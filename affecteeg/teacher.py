"""Extract frozen MD-JPT features; fitted targets belong to the training fold."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .common import read_json, write_json, resolve_device


def _file_version(path):
    """Cheap cache invalidation for immutable prepared inputs, without rereading EEG."""
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _extract(array, channels, source_paths, destination, *, kind, device, batch_size):
    from .mdjpt import FrozenMdJPTEncoder, CHECKPOINT_PATH, CONFIG_PATH, INTERPOLATION_PATH
    from .mdjpt import per_channel_zscore, resample_poly_windows
    destination = Path(destination)
    metadata_path = destination.with_suffix(".json")
    identity = {
        "format": 1, "kind": kind, "shape": [len(array), 512], "channels": channels,
        "inputs": [_file_version(p) for p in source_paths],
        "encoder": [_file_version(p) for p in (CHECKPOINT_PATH, CONFIG_PATH, INTERPOLATION_PATH)],
        "preprocessing": "prepared_125Hz_500_samples" if kind == "tone" else "per_trial_channel_zscore_polyphase_200_to_125Hz",
    }
    if destination.is_file() and metadata_path.is_file() and read_json(metadata_path) == identity:
        existing = np.load(destination, mmap_mode="r", allow_pickle=False)
        if existing.shape == (len(array), 512) and existing.dtype == np.float32:
            return destination
    if int(batch_size) <= 0:
        raise ValueError("MD-JPT batch size must be positive")
    encoder = FrozenMdJPTEncoder().to(resolve_device(device)).eval()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    output = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=(len(array), 512))
    try:
        for start in range(0, len(array), int(batch_size)):
            batch = np.asarray(array[start:start + batch_size], dtype=np.float32)
            if not np.isfinite(batch).all():
                raise ValueError(f"Nonfinite teacher EEG at row {start}")
            if kind != "tone":
                batch = resample_poly_windows(per_channel_zscore(batch), source_rate=200)
            if batch.shape[1:] != (500, len(channels)):
                raise ValueError(f"Teacher expects 500 time points and explicit channels, got {batch.shape}")
            output[start:start + len(batch)] = encoder.encode_numpy(
                batch, channels, device=resolve_device(device), batch_size=batch_size, amp=False)
        output.flush()
        del output
        output = None
        temporary.replace(destination)
        write_json(metadata_path, identity)
    except BaseException:
        if output is not None:
            del output
        temporary.unlink(missing_ok=True)
        raise
    return destination


def extract_tone_features(processed_root, output_root, device="cpu", batch_size=128):
    from .data import load_tone
    data = load_tone(processed_root)
    root = Path(processed_root) / "tone"
    return _extract(data["eeg"], data["metadata"]["channels"],
                    [root / "eeg.npy", root / "metadata.json"],
                    Path(output_root) / "teacher" / "tone" / "features.npy",
                    kind="tone", device=device, batch_size=batch_size)


def extract_content_features(processed_root, track, output_root, device="cpu", batch_size=128):
    from .data import load_cevb
    data = load_cevb(processed_root, track)
    root = Path(processed_root) / "cevb" / track
    return _extract(data["content_eeg"], data["manifest"]["content_channels"],
                    [root / "content_eeg.npy", root / "manifest.json"],
                    Path(output_root) / "teacher" / track / "features.npy",
                    kind="reconstruction", device=device, batch_size=batch_size)
