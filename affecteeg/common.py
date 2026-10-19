"""Small shared runtime utilities; all paths are supplied by the caller."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, default=_json_default, allow_nan=False) + "\n")
    temporary.replace(path)


def seed_all(seed):
    import numpy as np
    import torch
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(value):
    import torch
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available; choose --device cpu or install CUDA PyTorch.")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return str(device)


def expand_selection(value, available, label="selection"):
    if str(value) == "all":
        return list(available)
    lookup = {str(item): item for item in available}
    if str(value) not in lookup:
        raise ValueError(f"Unknown {label} {value!r}; choose from {list(lookup)} or all")
    return [lookup[str(value)]]
