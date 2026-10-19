"""Common channels-last model interface."""
from __future__ import annotations
import torch
from torch import nn

class ToneBaselineWrapper(nn.Module):
    """Adapt a faithful EEG model to the common EEG/EMG tuple interface."""

    def __init__(
        self,
        eeg_channels: int,
        emg_channels: int,
        num_classes: int,
        modality: str = "eeg",
    ) -> None:
        super().__init__()
        self.eeg_channels = eeg_channels
        self.emg_channels = emg_channels
        self.num_classes = num_classes
        self.modality = modality

    def raw_channels(
        self,
        signal: torch.Tensor,
        expected_channels: int | None = None,
    ) -> torch.Tensor:
        """Explicitly remove the appended first-difference channel block."""
        raw_count = self.eeg_channels if expected_channels is None else expected_channels
        if signal.ndim != 3 or signal.shape[-1] not in {raw_count, raw_count * 2}:
            raise ValueError(
                f"expected (B,T,{raw_count}) raw or (B,T,{raw_count * 2}) raw+diff "
                f"input, got {tuple(signal.shape)}"
            )
        return signal if signal.shape[-1] == raw_count else signal[..., :raw_count]

    def classify(self, eeg: torch.Tensor, emg: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, eeg: torch.Tensor, emg: torch.Tensor):
        raw_eeg = (
            self.raw_channels(eeg, self.eeg_channels)
            if self.modality in {"eeg", "eeg_emg"}
            else eeg
        )
        raw_emg = (
            self.raw_channels(emg, self.emg_channels)
            if self.modality in {"emg", "eeg_emg"}
            else emg
        )
        class_logits = self.classify(raw_eeg, raw_emg)
        domain_placeholder = class_logits.new_zeros((class_logits.shape[0], 1))
        return class_logits, domain_placeholder
