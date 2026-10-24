"""Dense PyTorch port of the official RGNN SGConv model."""

from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ToneBaselineWrapper


FIVE_CHANNEL_ADJACENCY = torch.tensor(
    [
        [1.0, -1.0, 0.20, 0.00, 0.00],
        [-1.0, 1.0, 0.00, 0.00, 0.20],
        [0.20, 0.00, 1.0, 0.80, 0.00],
        [0.00, 0.00, 0.80, 1.0, 0.80],
        [0.00, 0.20, 0.00, 0.80, 1.0],
    ],
    dtype=torch.float32,
)

FIVE_CHANNEL_GRAPH_METADATA = {
    "channels": ("T7", "T8", "Fp1", "Fpz", "Fp2"),
    "topology": "approved five-edge bilateral/nearest-neighbour task graph",
    "edges": (
        ("T7", "T8", -1.0, "bilateral differential-asymmetry link"),
        ("T7", "Fp1", 0.2, "left temporal-frontal task link"),
        ("T8", "Fp2", 0.2, "right temporal-frontal task link"),
        ("Fp1", "Fpz", 0.8, "frontal nearest-neighbour link"),
        ("Fpz", "Fp2", 0.8, "frontal nearest-neighbour link"),
    ),
    "negative_edges": (("T7", "T8"),),
    "derived_from_seed_62": False,
}


STANDARD_COORDINATE_SOURCE = (
    Path(__file__).resolve().parents[3] / "vendor/tone/channel_location.py"
)
# LibEER/data_utils/constants/seed.py: SEED_GLOBAL_CHANNEL_PAIRS.
STANDARD_BILATERAL_PAIRS = (
    ("FP1", "FP2"), ("AF3", "AF4"), ("F5", "F6"),
    ("FC5", "FC6"), ("C5", "C6"), ("CP5", "CP6"),
    ("P5", "P6"), ("PO5", "PO6"), ("O1", "O2"),
)


@lru_cache(maxsize=1)
def _standard_coordinates() -> dict[str, list[float]]:
    """Read the checked-in 10-05 coordinates."""
    spec = importlib.util.spec_from_file_location(
        "_rgnn_standard_coordinates", STANDARD_COORDINATE_SOURCE
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load RGNN electrode coordinates: {STANDARD_COORDINATE_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.system_10_05_loc


def standard_coordinate_adjacency(channel_names: tuple[str, ...]) -> torch.Tensor:
    """Apply LibEER's SEED RGNN distance/asymmetry rule to named electrodes.

    Its millimetre coordinates are divided by 10, then off-diagonal weights
    are min(5 / squared_distance_cm, 1). Present bilateral pairs lose one in
    both directions. Electrode names and coordinates determine the graph.
    """
    names = tuple(name.upper() for name in channel_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("RGNN channel_names must contain unique electrode names")
    coordinates = _standard_coordinates()
    missing = [name for name in names if name not in coordinates]
    if missing:
        raise ValueError(f"RGNN channel_names have no standard coordinates: {missing}")
    positions = torch.tensor([coordinates[name] for name in names], dtype=torch.float64) / 10
    distance_squared = (positions[:, None] - positions[None, :]).square().sum(-1)
    adjacency = (5 / distance_squared).clamp_max(1)
    adjacency.fill_diagonal_(1)
    lookup = {name: index for index, name in enumerate(names)}
    for left, right in STANDARD_BILATERAL_PAIRS:
        if left in lookup and right in lookup:
            i, j = lookup[left], lookup[right]
            adjacency[i, j] -= 1
            adjacency[j, i] -= 1
    return adjacency.to(torch.float32)


def normalized_signed_adjacency(adjacency: torch.Tensor) -> torch.Tensor:
    """Official NewSGConv normalization uses absolute weights for degree."""
    degree = adjacency.abs().sum(dim=-1)
    inverse_sqrt = degree.clamp_min(torch.finfo(adjacency.dtype).eps).rsqrt()
    return inverse_sqrt[:, None] * adjacency * inverse_sqrt[None, :]


def dense_sgconv_reference(
    x: torch.Tensor, adjacency: torch.Tensor, k: int
) -> torch.Tensor:
    normalized = normalized_signed_adjacency(adjacency)
    for _ in range(k):
        x = torch.einsum("ij,bjf->bif", normalized, x)
    return x


class DenseSignedSGConv(nn.Module):
    """Dense equivalent of official NewSGConv for a named-electrode graph."""

    def __init__(self, in_features: int, out_features: int, k: int = 2) -> None:
        super().__init__()
        self.k = k
        self.linear = nn.Linear(in_features, out_features)

    def propagate(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        return dense_sgconv_reference(x, adjacency, self.k)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        return self.linear(self.propagate(x, adjacency))


class FaithfulRGNN(ToneBaselineWrapper):
    """Official signed SGConv, graph sum pooling, dropout, and L1 graph loss."""

    port_notes = (
        "Dense PyTorch SGConv with the five-channel Tone5 graph; "
        "larger montages use 10-05 coordinate RGNN initialization."
    )
    graph_metadata = FIVE_CHANNEL_GRAPH_METADATA

    def __init__(
        self,
        timesteps: int,
        eeg_channels: int,
        emg_channels: int,
        num_classes: int,
        init_mode: str = "scratch",
        channel_names: tuple[str, ...] = ("T7", "T8", "Fp1", "Fpz", "Fp2"),
    ) -> None:
        super().__init__(eeg_channels, emg_channels, num_classes)
        if init_mode != "scratch":
            raise ValueError("RGNN supports init_mode='scratch' only")
        if eeg_channels < 2 or timesteps < 1:
            raise ValueError("RGNN requires at least two EEG channels and T >= 1")
        expected_names = FIVE_CHANNEL_GRAPH_METADATA["channels"]
        channel_names = tuple(channel_names)
        if len(channel_names) != eeg_channels:
            raise ValueError("RGNN channel_names must match eeg_channels")
        if eeg_channels == 5 and set(channel_names) == set(expected_names):
            adjacency = FIVE_CHANNEL_ADJACENCY
            node_names = expected_names
            self.graph_metadata = FIVE_CHANNEL_GRAPH_METADATA
        else:
            adjacency = standard_coordinate_adjacency(channel_names)
            node_names = channel_names
            self.graph_metadata = {
                "channels": channel_names,
                "topology": "10-05 coordinate distance graph with RGNN bilateral asymmetry",
                "coordinate_source": str(STANDARD_COORDINATE_SOURCE),
                "initialization_source": "repos/LibEER/LibEER/data_utils/preprocess.py:generate_rgnn_adjacency_matrix",
                "distance_rule": "min(5 / squared_distance_cm, 1); diagonal=1",
                "bilateral_rule": "subtract 1 symmetrically when both electrodes are present",
                "bilateral_pairs": tuple(
                    pair for pair in STANDARD_BILATERAL_PAIRS
                    if set(pair) <= {name.upper() for name in channel_names}
                ),
                "derived_from_seed_62": False,
            }
        self.timesteps = timesteps
        self.channel_names = channel_names
        reorder = torch.tensor(
            [channel_names.index(name) for name in node_names], dtype=torch.long
        )
        self.register_buffer("channel_reorder", reorder)
        xs, ys = torch.tril_indices(eeg_channels, eeg_channels)
        self.register_buffer("lower_rows", xs)
        self.register_buffer("lower_cols", ys)
        self.edge_weight = nn.Parameter(adjacency[xs, ys].clone())
        self.graph_conv = DenseSignedSGConv(timesteps, 64, k=2)
        self.classifier = nn.Linear(64, num_classes)
        self.dropout = 0.7

    def adjacency_matrix(self) -> torch.Tensor:
        adjacency = self.edge_weight.new_zeros((self.eeg_channels, self.eeg_channels))
        adjacency[self.lower_rows, self.lower_cols] = self.edge_weight
        return adjacency + adjacency.T - torch.diag(adjacency.diagonal())

    def classify(self, eeg: torch.Tensor, emg: torch.Tensor) -> torch.Tensor:
        del emg
        if eeg.shape[1] != self.timesteps:
            raise ValueError(f"RGNN expected {self.timesteps} timesteps, got {eeg.shape[1]}")
        nodes = eeg[:, :, self.channel_reorder].transpose(1, 2)  # (B, C, T)
        nodes = F.relu(self.graph_conv(nodes, self.adjacency_matrix()))
        graph = F.dropout(nodes.sum(dim=1), self.dropout, self.training)
        return self.classifier(graph)

    def regularization_loss(self) -> torch.Tensor:
        return 0.01 * self.edge_weight.abs().sum()
