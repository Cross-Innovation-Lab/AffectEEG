"""PyTorch port of the official EEG-ATCNet Keras ``ATCNet_`` architecture."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ToneBaselineWrapper


def _glorot_uniform_(
    module: nn.Conv1d | nn.Conv2d | nn.Linear,
    keras_fans: tuple[int, int] | None = None,
) -> None:
    if keras_fans is None:
        nn.init.xavier_uniform_(module.weight)
    else:
        bound = (6.0 / sum(keras_fans)) ** 0.5
        nn.init.uniform_(module.weight, -bound, bound)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    module.initializer_kind = "glorot_uniform"


def _keras_einsum_fans(kernel_shape: tuple[int, ...]) -> tuple[int, int]:
    if len(kernel_shape) < 2:
        raise ValueError("Keras EinsumDense kernel shape must have at least 2 axes")
    receptive_field = math.prod(kernel_shape[:-2])
    return (
        kernel_shape[-2] * receptive_field,
        kernel_shape[-1] * receptive_field,
    )


def _keras_einsum_glorot_uniform_(
    weight: torch.Tensor,
    kernel_shape: tuple[int, ...],
) -> None:
    fan_in, fan_out = _keras_einsum_fans(kernel_shape)
    bound = (6.0 / (fan_in + fan_out)) ** 0.5
    nn.init.uniform_(weight, -bound, bound)


def _he_uniform_(module: nn.Conv1d) -> None:
    nn.init.kaiming_uniform_(module.weight, a=0.0, mode="fan_in", nonlinearity="relu")
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    module.initializer_kind = "he_uniform"


def _constraint_norms(
    kind: str,
    weight: torch.Tensor,
    depth_multiplier: int = 2,
) -> torch.Tensor:
    if kind == "depthwise":
        grouped = weight.reshape(-1, depth_multiplier, weight[0].numel())
        return grouped.permute(1, 0, 2).flatten(1).norm(dim=1)
    return weight.flatten(1).norm(dim=1)


@torch.no_grad()
def _apply_max_norm(
    kind: str,
    weight: torch.Tensor,
    limit: float,
    depth_multiplier: int = 2,
) -> None:
    norms = _constraint_norms(kind, weight, depth_multiplier)
    scales = (limit / norms.clamp_min(1e-12)).clamp(max=1.0)
    if kind == "depthwise":
        grouped = weight.reshape(-1, depth_multiplier, weight[0].numel())
        grouped.mul_(scales.reshape(1, depth_multiplier, 1))
    else:
        weight.mul_(scales.reshape(-1, *([1] * (weight.ndim - 1))))


class _CausalConv1d(nn.Module):
    def __init__(self, inputs: int, outputs: int, kernel: int, dilation: int) -> None:
        super().__init__()
        self.left_padding = dilation * (kernel - 1)
        self.conv = nn.Conv1d(inputs, outputs, kernel, dilation=dilation)
        _he_uniform_(self.conv)
        self.initializer_kind = self.conv.initializer_kind

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.left_padding, 0)))


class _TCNUnit(nn.Module):
    def __init__(
        self,
        input_dimension: int,
        filters: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        project_residual: bool,
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            _CausalConv1d(input_dimension, filters, kernel_size, dilation),
            nn.BatchNorm1d(filters, eps=1e-3, momentum=0.01),
            nn.ELU(),
            nn.Dropout(dropout),
            _CausalConv1d(filters, filters, kernel_size, dilation),
            nn.BatchNorm1d(filters, eps=1e-3, momentum=0.01),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        self.input_projection = (
            nn.Conv1d(input_dimension, filters, 1) if project_residual else None
        )
        if self.input_projection is not None:
            _glorot_uniform_(self.input_projection)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels_first = x.transpose(1, 2)
        residual = channels_first
        if self.input_projection is not None:
            residual = self.input_projection(residual)
        return F.elu(self.layers(channels_first) + residual).transpose(1, 2)


class _TCNBlock(nn.Module):
    def __init__(
        self,
        input_dimension: int,
        filters: int = 32,
        depth: int = 2,
        kernel_size: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("TCN depth must be at least 1")
        first_projection = input_dimension != filters
        units = [
            _TCNUnit(
                input_dimension, filters, kernel_size, 1, dropout, first_projection
            )
        ]
        units.extend(
            _TCNUnit(filters, filters, kernel_size, 2**level, dropout, False)
            for level in range(1, depth)
        )
        self.units = nn.ModuleList(units)

    @property
    def input_projection(self) -> nn.Conv1d | None:
        return self.units[0].input_projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for unit in self.units:
            x = unit(x)
        return x


class _OfficialMHA(nn.Module):
    def __init__(self, features: int, key_dim: int = 8, heads: int = 2) -> None:
        super().__init__()
        inner = key_dim * heads
        self.key_dim = key_dim
        self.heads = heads
        self.norm = nn.LayerNorm(features, eps=1e-6)
        self.query = nn.Linear(features, inner)
        self.key = nn.Linear(features, inner)
        self.value = nn.Linear(features, inner)
        self.attention_dropout = nn.Dropout(0.5)
        self.output = nn.Linear(inner, features)
        self.output_dropout = nn.Dropout(0.3)
        query_kernel_shape = (features, heads, key_dim)
        output_kernel_shape = (heads, key_dim, features)
        for projection in (self.query, self.key, self.value):
            _keras_einsum_glorot_uniform_(projection.weight, query_kernel_shape)
            nn.init.zeros_(projection.bias)
            projection.initializer_kind = "keras_einsum_glorot_uniform"
        _keras_einsum_glorot_uniform_(self.output.weight, output_kernel_shape)
        nn.init.zeros_(self.output.bias)
        self.output.initializer_kind = "keras_einsum_glorot_uniform"

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        return x.reshape(batch, length, self.heads, self.key_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(x)
        query = self._split_heads(self.query(normalized))
        key = self._split_heads(self.key(normalized))
        value = self._split_heads(self.value(normalized))
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.key_dim**0.5
        weights = self.attention_dropout(F.softmax(scores, dim=-1))
        attended = torch.matmul(weights, value).transpose(1, 2).flatten(2)
        return x + self.output_dropout(self.output(attended))


class _ATCNet(nn.Module):
    def __init__(self, electrodes: int, timesteps: int, classes: int) -> None:
        super().__init__()
        self.num_windows = 5
        self.depth_multiplier = 2
        self.initialization_scheme = {
            "conv2d": "glorot_uniform",
            "depthwise_conv2d": "glorot_uniform",
            "tcn_conv1d": "he_uniform",
            "residual_projection": "glorot_uniform",
            "dense": "glorot_uniform",
            "mha_einsum": "keras_einsum_glorot_uniform",
            "bias": "zeros",
        }
        self.feature_timesteps = (timesteps // 8) // 7
        if self.feature_timesteps < self.num_windows:
            raise ValueError(
                f"ATCNet needs at least {self.num_windows} pooled timesteps, "
                f"got {self.feature_timesteps} from {timesteps}"
            )
        window_length = self.feature_timesteps - self.num_windows + 1
        self.window_bounds = tuple(
            (index, index + window_length) for index in range(self.num_windows)
        )
        self.conv_block = self._conv_block(electrodes)
        self.attention = nn.ModuleList([_OfficialMHA(32) for _ in range(5)])
        self.tcns = nn.ModuleList([_TCNBlock(32) for _ in range(5)])
        self.classifiers = nn.ModuleList([nn.Linear(32, classes) for _ in range(5)])
        for classifier in self.classifiers:
            _glorot_uniform_(classifier)

    @staticmethod
    def _conv_block(electrodes: int) -> nn.Sequential:
        temporal = nn.Conv2d(1, 16, (1, 64), padding="same", bias=False)
        depthwise = nn.Conv2d(16, 32, (electrodes, 1), groups=16, bias=False)
        spatial = nn.Conv2d(32, 32, (1, 16), padding="same", bias=False)
        _glorot_uniform_(temporal)
        _glorot_uniform_(depthwise, keras_fans=(electrodes * 16, electrodes * 2))
        _glorot_uniform_(spatial)
        return nn.Sequential(
            temporal,
            nn.BatchNorm2d(16, eps=1e-3, momentum=0.01),
            depthwise,
            nn.BatchNorm2d(32, eps=1e-3, momentum=0.01),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.3),
            spatial,
            nn.BatchNorm2d(32, eps=1e-3, momentum=0.01),
            nn.ELU(),
            nn.AvgPool2d((1, 7)),
            nn.Dropout(0.3),
        )

    def conv_tcn_weights(self) -> list[torch.Tensor]:
        weights = [
            self.conv_block[0].weight,
            self.conv_block[2].weight,
            self.conv_block[7].weight,
        ]
        for tcn in self.tcns:
            for unit in tcn.units:
                weights.extend(
                    layer.conv.weight
                    for layer in unit.layers
                    if isinstance(layer, _CausalConv1d)
                )
                if unit.input_projection is not None:
                    weights.append(unit.input_projection.weight)
        return weights

    def constrained_parameters(self) -> list[tuple[str, torch.Tensor]]:
        constrained = [
            ("output", self.conv_block[0].weight),
            ("depthwise", self.conv_block[2].weight),
            ("output", self.conv_block[7].weight),
        ]
        conv_weights = self.conv_tcn_weights()[3:]
        constrained.extend(("output", weight) for weight in conv_weights)
        return constrained

    def _window_logits(self, features: torch.Tensor, index: int) -> torch.Tensor:
        start, end = self.window_bounds[index]
        window = self.attention[index](features[:, start:end, :])
        encoded = self.tcns[index](window)
        return self.classifiers[index](encoded[:, -1, :])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv_block(x).squeeze(2).transpose(1, 2)
        logits = [
            self._window_logits(features, index) for index in range(self.num_windows)
        ]
        return torch.stack(logits).mean(dim=0)


class FaithfulATCNet(ToneBaselineWrapper):
    """Channels-last adapter for the official Keras ATCNet_ PyTorch port."""

    def __init__(
        self,
        timesteps: int,
        eeg_channels: int,
        emg_channels: int,
        num_classes: int,
        init_mode: str = "scratch",
    ) -> None:
        super().__init__(eeg_channels, emg_channels, num_classes)
        if init_mode != "scratch":
            raise ValueError("ATCNet supports init_mode='scratch' only")
        self.input_timesteps = timesteps
        self.temporal_adjustment = "none"
        self.port_notes = (
            "PyTorch port of official Keras ATCNet_ at "
            "EEG-ATCNet/models.py; TorchEEG line 173 bug is not copied. "
            "MHA output feeds dropout and residual; TCN projection is conditional. "
            "Training hooks preserve conv/TCN L2(0.009), classifier L2(0.5), "
            "and kernel max_norm(0.6). MHA initialization uses Keras "
            "EinsumDense three-axis kernel fans."
        )
        self.backbone = _ATCNet(eeg_channels, timesteps, num_classes)

    def regularized_parameter_groups(self) -> dict[str, list[torch.Tensor]]:
        return {
            "conv_tcn": self.backbone.conv_tcn_weights(),
            "classifier": [layer.weight for layer in self.backbone.classifiers],
        }

    def regularization_loss(self) -> torch.Tensor:
        groups = self.regularized_parameter_groups()
        conv_l2 = sum(weight.square().sum() for weight in groups["conv_tcn"])
        classifier_l2 = sum(weight.square().sum() for weight in groups["classifier"])
        return 0.009 * conv_l2 + 0.5 * classifier_l2

    def constrained_parameters(self) -> list[tuple[str, torch.Tensor]]:
        return self.backbone.constrained_parameters()

    @torch.no_grad()
    def apply_constraints(self) -> None:
        for kind, weight in self.constrained_parameters():
            _apply_max_norm(kind, weight, 0.6, self.backbone.depth_multiplier)

    def constraint_norms(self) -> list[float]:
        norms = [
            _constraint_norms(kind, weight, self.backbone.depth_multiplier)
            for kind, weight in self.constrained_parameters()
        ]
        return torch.cat(norms).detach().cpu().tolist()

    def classify(self, eeg: torch.Tensor, emg: torch.Tensor) -> torch.Tensor:
        del emg
        if eeg.shape[1] != self.input_timesteps:
            raise ValueError(f"expected {self.input_timesteps} timesteps, got {eeg.shape[1]}")
        # The official Keras model receives (B, 1, electrodes, time).
        eeg_2d = eeg.permute(0, 2, 1).unsqueeze(1)
        return self.backbone(eeg_2d)
