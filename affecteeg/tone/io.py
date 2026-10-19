"""Local checkpoint I/O for checkpoints created by this package."""
from pathlib import Path
import os
import torch


def save_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(path):
    # These local training artifacts include NumPy protocol arrays.
    return torch.load(path, map_location='cpu', weights_only=False)
