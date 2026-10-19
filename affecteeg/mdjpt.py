"""Frozen MD-JPT inference with local weights and explicit channel alignment.

Adapted from the project encoder and ncclab-sustech/mdJPT_nips2025.
No pretraining pipeline or online model loading is required.
"""
from pathlib import Path
from typing import Sequence
import json
import numpy as np
import torch
from torch import nn
from scipy.signal import resample_poly
from ._vendor.mdjpt.MLLA_new import channel_MLLA
from ._vendor.mdjpt.Channel_MLP import Channel_mlp_CNN
from ._vendor.mdjpt.CNN_Attention import cnn_MLLA

ASSETS = Path(__file__).resolve().parents[1] / "assets" / "mdjpt"
CHECKPOINT_PATH = ASSETS / "epoch=19.ckpt"
CONFIG_PATH = ASSETS / "config.json"
INTERPOLATION_PATH = ASSETS / "channel_interpolate.npy"
MDJPT_FEATURE_DIM = 512
MDJPT_TARGET_RATE = 125


def _require_file(path, label):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"MD-JPT {label} is missing: {path}")
    return path


class _FrozenBackbone(nn.Module):
    """Exact checkpoint-bearing inference modules, without Lightning training code."""
    def __init__(self, cfg):
        super().__init__()
        self.uni_mlp = Channel_mlp_CNN(len(cfg["uni_channels"]), cfg["cnn"]["n_channs"])
        self.MLLA = channel_MLLA(**{name: cfg[name] for name in (
            "context_window", "patch_size", "hidden_dim", "out_dim", "depth", "patch_stride", "n_heads")})
        self.cnn_encoder = cnn_MLLA(**cfg["cnn"])


def _rowwise_stratified_layer_norm(value: torch.Tensor) -> torch.Tensor:
    """Vectorized equivalent of vendor ``stratified_layerNorm(..., n_samples=1)``."""
    if value.ndim != 4 or value.shape[-1] < 2:
        raise ValueError("rowwise mdJPT normalization requires [B,D1,D2,T] with T>=2")
    batch, dim1, dim2, timesteps = value.shape
    flattened = value.permute(0, 3, 1, 2).reshape(batch, timesteps, dim1 * dim2)
    mean = flattened.mean(dim=1, keepdim=True)
    std = flattened.std(dim=1, keepdim=True, correction=1)
    normalized = (flattened - mean) / (std + 1.0e-3)
    return normalized.reshape(batch, timesteps, dim1, dim2).permute(0, 2, 3, 1)

class FrozenMdJPTEncoder(nn.Module):
    """Strictly loaded, fully frozen, batch-invariant mdJPT feature encoder.

    Inputs use ``[batch, time, channels]`` and explicit channel names.  The adapter
    keeps variable-duration inputs native; it does not silently pad or time-warp.
    Sampling-rate conversion and dataset-specific normalization are caller-owned.
    """

    output_dim = MDJPT_FEATURE_DIM
    target_sample_rate = MDJPT_TARGET_RATE

    def __init__(
        self,
        checkpoint_path: str | Path = CHECKPOINT_PATH,
        config_path: str | Path = CONFIG_PATH,
        channel_map_path: str | Path = INTERPOLATION_PATH,
    ) -> None:
        super().__init__()
        self.checkpoint_path = _require_file(Path(checkpoint_path), "checkpoint")
        self.config_path = _require_file(Path(config_path), "configuration")
        self.channel_map_path = _require_file(
            Path(channel_map_path), "channel interpolation table"
        )
        cfg = json.loads(self.config_path.read_text())
        backbone = _FrozenBackbone(cfg)
        checkpoint = torch.load(
            self.checkpoint_path, map_location="cpu", weights_only=True
        )
        state_dict = checkpoint.get("state_dict") if isinstance(checkpoint, dict) else None
        if not isinstance(state_dict, dict):
            raise RuntimeError("mdJPT checkpoint does not contain a state_dict")
        backbone.load_state_dict(state_dict, strict=True)
        backbone.save_fea = True
        backbone.cnn_encoder.set_saveFea(True)
        # The adapter applies the exact n_samples=1 equivalent itself.  Disabling
        # the vendor call removes its B//2 batch-composition dependency.
        backbone.cnn_encoder.set_stratified([])
        backbone.requires_grad_(False)
        backbone.eval()
        self.backbone = backbone
        self.uni_channels = tuple(str(name) for name in cfg["uni_channels"])
        interpolation = np.load(self.channel_map_path, allow_pickle=False).astype(np.int64)
        if interpolation.shape != (60, 60):
            raise RuntimeError("mdJPT channel interpolation shape drift")
        self.register_buffer(
            "channel_interpolation",
            torch.from_numpy(interpolation),
            persistent=False,
        )
        self.eval()

    def train(self, mode: bool = True):  # type: ignore[override]
        del mode
        return super().train(False)

    def _align_channels(
        self, windows: torch.Tensor, channel_names: Sequence[str]
    ) -> torch.Tensor:
        if windows.ndim != 3:
            raise ValueError("mdJPT input must have shape [B,T,C]")
        names = tuple(str(name).upper() for name in channel_names)
        if len(names) != windows.shape[-1]:
            raise ValueError("mdJPT channel-name count does not match input")
        if len(set(names)) != len(names):
            raise ValueError("mdJPT input contains duplicate channel names")
        source = {name: index for index, name in enumerate(names)}
        by_channel = windows.permute(0, 2, 1)
        aligned: list[torch.Tensor] = []
        for standard_index, standard_name in enumerate(self.uni_channels):
            direct = source.get(standard_name.upper())
            if direct is not None:
                aligned.append(by_channel[:, direct])
                continue
            neighbors: list[int] = []
            for neighbor_index in self.channel_interpolation[standard_index].tolist():
                neighbor_name = self.uni_channels[int(neighbor_index)].upper()
                source_index = source.get(neighbor_name)
                if source_index is not None and source_index not in neighbors:
                    neighbors.append(source_index)
                if len(neighbors) == 3:
                    break
            if not neighbors:
                raise ValueError(
                    f"mdJPT cannot align channel {standard_name}: no registered neighbor"
                )
            aligned.append(by_channel[:, neighbors].mean(dim=1))
        return torch.stack(aligned, dim=1).unsqueeze(1)

    def _mlla_forward(self, projected: torch.Tensor) -> torch.Tensor:
        batch, singleton, channels, timesteps = projected.shape
        if singleton != 1 or timesteps < int(self.backbone.MLLA.patch_size):
            raise ValueError("mdJPT aligned input has an invalid shape")
        patches = projected[:, 0].unfold(
            dimension=-1,
            size=int(self.backbone.MLLA.patch_size),
            step=int(self.backbone.MLLA.patch_stride),
        )
        patch_count = patches.shape[2]
        encoded = self.backbone.MLLA.encoder(
            patches.reshape(batch * channels, patch_count, -1)
        )
        encoded = encoded.reshape(batch, channels, patch_count, encoded.shape[-1])
        encoded = encoded.permute(0, 1, 3, 2)
        encoded = _rowwise_stratified_layer_norm(encoded)
        return encoded.permute(0, 1, 3, 2)

    def forward(
        self,
        windows: torch.Tensor,
        channel_names: Sequence[str],
        *,
        return_feature_map: bool = False,
    ) -> torch.Tensor:
        if any(parameter.requires_grad for parameter in self.backbone.parameters()):
            raise RuntimeError("mdJPT backbone is not fully frozen")
        windows = windows.to(dtype=torch.float32)
        projected = self._align_channels(windows, channel_names)
        mlla = self._mlla_forward(projected)
        covariance = self.backbone.uni_mlp(mlla.permute(0, 3, 1, 2))
        covariance = _rowwise_stratified_layer_norm(covariance)
        feature_map = self.backbone.cnn_encoder(covariance)
        expected = (windows.shape[0], MDJPT_FEATURE_DIM, 1)
        if feature_map.shape[:3] != expected or not torch.isfinite(feature_map).all():
            raise RuntimeError(f"mdJPT feature-map contract failed: {tuple(feature_map.shape)}")
        if return_feature_map:
            return feature_map
        features = feature_map.mean(dim=-1).squeeze(-1)
        if features.shape != (windows.shape[0], MDJPT_FEATURE_DIM):
            raise RuntimeError("mdJPT mean-feature shape drift")
        return features

    @torch.inference_mode()
    def encode_numpy(
        self,
        windows: np.ndarray,
        channel_names: Sequence[str],
        *,
        device: str | torch.device,
        batch_size: int = 64,
        amp: bool = False,
    ) -> np.ndarray:
        array = np.asarray(windows, dtype=np.float32)
        if array.ndim != 3:
            raise ValueError("mdJPT numpy input must have shape [N,T,C]")
        target = torch.device(device)
        self.to(target).eval()
        rows: list[np.ndarray] = []
        for start in range(0, len(array), int(batch_size)):
            batch = torch.from_numpy(np.array(array[start : start + batch_size], copy=True, order="C"))
            batch = batch.to(target, non_blocking=target.type == "cuda")
            with torch.amp.autocast(
                target.type,
                enabled=bool(amp and target.type == "cuda"),
            ):
                encoded = self(batch, channel_names)
            rows.append(encoded.float().cpu().numpy())
        result = np.concatenate(rows, axis=0).astype(np.float32, copy=False)
        if result.shape != (len(array), MDJPT_FEATURE_DIM) or not np.isfinite(result).all():
            raise RuntimeError("mdJPT numpy encoding contract failed")
        return result

def per_channel_zscore(windows: np.ndarray, *, epsilon: float = 1.0e-6) -> np.ndarray:
    values = np.asarray(windows, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("trialwise z-score expects [N,T,C]")
    mean = values.mean(axis=1, keepdims=True, dtype=np.float64).astype(np.float32)
    std = values.std(axis=1, keepdims=True, dtype=np.float64).astype(np.float32)
    normalized = (values - mean) / (std + float(epsilon))
    return np.asarray(normalized, dtype=np.float32)

def resample_poly_windows(
    windows: np.ndarray,
    *,
    source_rate: int,
    target_rate: int = MDJPT_TARGET_RATE,
) -> np.ndarray:
    values = np.asarray(windows, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("polyphase resampling expects [N,T,C]")
    divisor = int(np.gcd(int(source_rate), int(target_rate)))
    up = int(target_rate) // divisor
    down = int(source_rate) // divisor
    result = resample_poly(values, up, down, axis=1)
    return np.asarray(result, dtype=np.float32)
