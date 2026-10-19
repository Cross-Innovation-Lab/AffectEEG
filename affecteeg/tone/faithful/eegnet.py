"""Faithful EEGNet wrapper backed by the checked-in TorchEEG implementation."""

from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path

import torch

from .base import ToneBaselineWrapper


def _local_eegnet_class():
    from vendor.tone.eegnet import EEGNet
    return EEGNet


class FaithfulEEGNet(ToneBaselineWrapper):
    """Channels-last adapter for TorchEEG's faithful Lawhern et al. EEGNet."""

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
            raise ValueError("EEGNet supports init_mode='scratch' only")
        eegnet_cls = _local_eegnet_class()
        self.backbone = eegnet_cls(
            chunk_size=timesteps,
            num_electrodes=eeg_channels,
            num_classes=num_classes,
        )

    def classify(self, eeg: torch.Tensor, emg: torch.Tensor) -> torch.Tensor:
        del emg
        # TorchEEG EEGNet consumes (B, 1, electrodes, time).
        eeg_2d = eeg.permute(0, 2, 1).unsqueeze(1)
        return self.backbone(eeg_2d)
