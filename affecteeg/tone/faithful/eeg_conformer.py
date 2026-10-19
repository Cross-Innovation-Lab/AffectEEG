"""Faithful EEG-Conformer extracted from the checked-in TorchEEG source."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ToneBaselineWrapper


class _PatchEmbedding(nn.Module):
    def __init__(self, electrodes: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.shallownet = nn.Sequential(
            nn.Conv2d(1, hidden, (1, 25)),
            nn.Conv2d(hidden, hidden, (electrodes, 1)),
            nn.BatchNorm2d(hidden),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),
            nn.Dropout(dropout),
        )
        self.projection = nn.Conv2d(hidden, hidden, (1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(self.shallownet(x))
        return x.flatten(2).transpose(1, 2)


class _MultiHeadAttention(nn.Module):
    def __init__(self, hidden: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.hidden = hidden
        self.heads = heads
        self.keys = nn.Linear(hidden, hidden)
        self.queries = nn.Linear(hidden, hidden)
        self.values = nn.Linear(hidden, hidden)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(hidden, hidden)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = x.shape
        return x.reshape(batch, tokens, self.heads, -1).permute(0, 2, 1, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        queries = self._heads(self.queries(x))
        keys = self._heads(self.keys(x))
        values = self._heads(self.values(x))
        energy = torch.einsum("bhqd,bhkd->bhqk", queries, keys)
        attention = self.att_drop(F.softmax(energy / self.hidden**0.5, dim=-1))
        output = torch.einsum("bhqk,bhkd->bhqd", attention, values)
        return self.projection(output.transpose(1, 2).flatten(2))


class _ResidualAdd(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.module(x)


class _TransformerBlock(nn.Sequential):
    def __init__(self, hidden: int, heads: int, dropout: float) -> None:
        super().__init__(
            _ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(hidden),
                    _MultiHeadAttention(hidden, heads, dropout),
                    nn.Dropout(dropout),
                )
            ),
            _ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(hidden),
                    nn.Linear(hidden, hidden * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden * 4, hidden),
                    nn.Dropout(dropout),
                )
            ),
        )


class _Conformer(nn.Module):
    def __init__(self, electrodes: int, timesteps: int, classes: int) -> None:
        super().__init__()
        hidden = 40
        self.embedding = _PatchEmbedding(electrodes, hidden, 0.5)
        self.encoder = nn.Sequential(
            *[_TransformerBlock(hidden, 10, 0.5) for _ in range(6)]
        )
        with torch.no_grad():
            mock = self.encoder(self.embedding(torch.zeros(1, 1, electrodes, timesteps)))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(mock.numel(), 32 * 8),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(32 * 8, 32),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(32, classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(self.embedding(x)))


class FaithfulEEGConformer(ToneBaselineWrapper):
    """Channels-last adapter for TorchEEG's EEG-Conformer."""

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
            raise ValueError("EEG-Conformer supports init_mode='scratch' only")
        self.input_timesteps = timesteps
        self.temporal_adjustment = "none"
        self.backbone = _Conformer(eeg_channels, timesteps, num_classes)

    def classify(self, eeg: torch.Tensor, emg: torch.Tensor) -> torch.Tensor:
        del emg
        if eeg.shape[1] != self.input_timesteps:
            raise ValueError(f"expected {self.input_timesteps} timesteps, got {eeg.shape[1]}")
        # TorchEEG Conformer consumes (B, 1, electrodes, time).
        eeg_2d = eeg.permute(0, 2, 1).unsqueeze(1)
        return self.backbone(eeg_2d)
