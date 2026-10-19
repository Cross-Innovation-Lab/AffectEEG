"""Training-only statistics and relative residual adaptation equations."""
from __future__ import annotations
import math, random
from dataclasses import asdict, is_dataclass
from typing import Any, Mapping
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score
EPSILON = 1e-6
class ProjectionNotEstimable(ValueError): pass

def _metrics(labels, logits):
    predictions = np.asarray(logits).argmax(1)
    return dict(accuracy=float(accuracy_score(labels, predictions)),
        balanced_accuracy=float(balanced_accuracy_score(labels, predictions)),
        macro_f1=float(f1_score(labels, predictions, labels=[0,1,2,3], average='macro', zero_division=0)))

def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def _cpu_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}

def _equal(left: Any, right: Any) -> bool:
    if is_dataclass(left) or is_dataclass(right):
        return type(left) is type(right) and _equal(asdict(left), asdict(right))
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor) and left.dtype == right.dtype and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return isinstance(left, np.ndarray) and isinstance(right, np.ndarray) and left.dtype == right.dtype and np.array_equal(left, right)
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(_equal(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        return len(left) == len(right) and all(_equal(a, b) for a, b in zip(left, right))
    return bool(left == right)

def _metric_key(metrics: Mapping[str, Any]) -> tuple[float, float]:
    # With 480 balanced trials, real metric changes are much larger than this;
    # rounding only prevents floating summation noise from breaking exact ties.
    return round(float(metrics["balanced_accuracy"]), 12), round(float(metrics["macro_f1"]), 12)

def feature_statistics(target: np.ndarray, epsilon: float = EPSILON) -> dict[str, np.ndarray]:
    value = np.asarray(target, dtype=np.float32)
    if value.ndim not in (2, 3) or not np.isfinite(value).all():
        raise ValueError("target must be a finite trial-by-feature or trial-by-window-by-feature array")
    mean = value.mean(0, dtype=np.float64).astype(np.float32)
    sigma = value.std(0, dtype=np.float64).astype(np.float32)
    active = sigma > epsilon
    return {"mean": mean, "sigma": sigma, "active": active,
            "safe_sigma": np.where(active, sigma, 1.0).astype(np.float32)}

def teacher_targets(emotion: np.ndarray, labels: np.ndarray, epsilon: float = EPSILON) -> tuple[np.ndarray, dict[str, Any]]:
    value = np.asarray(emotion, dtype=np.float32)
    if value.ndim != 2 or len(value) != len(labels) or not np.isfinite(value).all():
        raise ValueError("teacher must be a finite matrix aligned to the training labels")
    if set(np.unique(labels)) != {0, 1, 2, 3}:
        raise ValueError("training partition must contain all four classes")
    means = np.stack([value[labels == tone].mean(0, dtype=np.float64) for tone in range(4)])
    residual = value.astype(np.float64) - means[labels]
    sigma = residual.std(0)
    active = sigma > epsilon
    normalized = (residual[:, active] / sigma[active]).astype(np.float32)
    return normalized, {"tone_means": means.astype(np.float32), "sigma": sigma.astype(np.float32),
                        "active": active, "epsilon": epsilon}

def tone_shuffle_indices(labels: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    permutation = np.arange(len(labels))
    for tone in range(4):
        rows = np.flatnonzero(labels == tone)
        permutation[rows] = rng.permutation(rows)
    return permutation

class RelativeAdapter(nn.Module):
    """Shared last-axis rank-16 adapter with per-position train-only scales."""
    def __init__(self, statistics: Mapping[str, np.ndarray], *, rank: int = 16, scale: float = 0.1):
        super().__init__()
        for name in ("mean", "sigma", "safe_sigma", "active"):
            self.register_buffer(name, torch.as_tensor(np.asarray(statistics[name]).copy()))
        dimension = int(self.mean.shape[-1])
        self.normalization = nn.LayerNorm(dimension)
        self.down = nn.Linear(dimension, rank)
        self.up = nn.Linear(rank, dimension)
        self.scale = float(scale)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def standardized(self, value: torch.Tensor) -> torch.Tensor:
        return (value - self.mean) / self.safe_sigma * self.active

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        standardized = self.standardized(value)
        delta = self.up(F.gelu(self.down(self.normalization(standardized))))
        return value - self.scale * self.sigma * self.active * torch.tanh(delta)

def standardized_view(clean: torch.Tensor, statistics: Mapping[str, torch.Tensor]) -> torch.Tensor:
    z = ((clean - statistics["mean"]) / statistics["safe_sigma"]).flatten(1)
    return z[:, statistics["active"].flatten()]

def covariance_penalty(clean_standardized: torch.Tensor, nuisance: torch.Tensor) -> torch.Tensor:
    if clean_standardized.shape[0] != nuisance.shape[0] or not clean_standardized.shape[1] or not nuisance.shape[1]:
        raise ProjectionNotEstimable("covariance has no active feature or teacher dimensions")
    x = clean_standardized - clean_standardized.mean(0, keepdim=True)
    z = nuisance - nuisance.mean(0, keepdim=True)
    return ((x.T @ z / len(x)).square()).mean()

def _head_regularizer(head: nn.Module, anchor: torch.Tensor) -> torch.Tensor:
    regularizer = getattr(head, "regularization_loss", None)
    return regularizer() if callable(regularizer) else anchor.new_zeros(())

def _norm(grads) -> float:
    return math.sqrt(sum(float(g.detach().float().square().sum()) for g in grads if g is not None))
