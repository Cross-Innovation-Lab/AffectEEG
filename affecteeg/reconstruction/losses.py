"""Masked shape loss and per-item original-log-mel evaluation metrics."""
from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
import torch
from torch.nn import functional as F


@dataclass
class MelStandardizer:
    mean: torch.Tensor
    std: torch.Tensor

    @classmethod
    def fit(cls, targets, train_keys):
        # Each key contributes exactly 100 training pairs, so one copy per key
        # has the same population moments and avoids loading duplicate targets.
        value = torch.as_tensor(np.asarray(targets[list(train_keys)]).copy(), dtype=torch.float64)
        return cls(value.mean((0, 2), keepdim=True).float(), value.std((0, 2), keepdim=True, unbiased=False).clamp_min(1e-6).float())

    def transform(self, value):
        return (value - self.mean.to(value.device)) / self.std.to(value.device)

    def inverse(self, value):
        return value * self.std.to(value.device) + self.mean.to(value.device)

    def state_dict(self):
        return {'mean': self.mean.cpu(), 'std': self.std.cpu()}


def pearson_loss(prediction, target, epsilon=1e-8):
    p, y = prediction.flatten(1), target.flatten(1)
    p, y = p - p.mean(1, keepdim=True), y - y.mean(1, keepdim=True)
    return 1 - ((p*y).sum(1)/(p.norm(dim=1)*y.norm(dim=1)).clamp_min(epsilon)).mean()


def corrected_shape_loss(prediction, target):
    energy = target.mean(1, keepdim=True)
    threshold = torch.quantile(energy.flatten(1), .2, dim=1, keepdim=True).unsqueeze(-1)
    mask = (energy >= threshold).to(target.dtype).expand_as(target)
    base = (F.smooth_l1_loss(prediction, target, beta=.5, reduction='none')*mask).sum()/mask.sum().clamp_min(1)
    p, y = prediction*mask, target*mask
    return (base + .5*torch.linalg.norm(p-y)/torch.linalg.norm(y).clamp_min(1e-8)
            + .2*F.l1_loss(p.diff(dim=-1), y.diff(dim=-1))
            + .1*F.l1_loss(p.diff(dim=1), y.diff(dim=1))
            + .5*pearson_loss(p, y)
            + .05*F.l1_loss(p.std((0, 2), unbiased=False), y.std((0, 2), unbiased=False)))


def reconstruction_loss(prediction, raw_target, scaler):
    shape = corrected_shape_loss(prediction, scaler.transform(raw_target))
    raw = pearson_loss(scaler.inverse(prediction), raw_target, epsilon=1e-12)
    return shape + .5*raw


def item_metrics(prediction, target):
    """PCC, mean absolute error, relative Frobenius error on 100 x 187."""
    p, y = prediction.double().flatten(1), target.double().flatten(1)
    pc, yc = p-p.mean(1, keepdim=True), y-y.mean(1, keepdim=True)
    pcc = (pc*yc).sum(1)/(pc.norm(dim=1)*yc.norm(dim=1)).clamp_min(1e-12)
    return torch.stack((pcc, (p-y).abs().mean(1), (p-y).norm(dim=1)/y.norm(dim=1).clamp_min(1e-12)), 1).detach().cpu().numpy()


def equal_metrics(rows):
    keys = defaultdict(list)
    for row in rows:
        values = np.array([row[k] for k in ('pcc', 'l1', 'rfe')], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError('nonfinite evaluation metrics')
        key = row['key']
        keys[(int(key.rsplit('|', 1)[1]), key)].append(values)
    if not keys:
        raise ValueError('empty evaluation support')
    emotions = defaultdict(list)
    for (emotion, _), values in keys.items():
        emotions[emotion].append(np.mean(values, axis=0))
    values = np.mean([np.mean(v, axis=0) for v in emotions.values()], axis=0)
    return dict(zip(('pcc', 'l1', 'rfe'), map(float, values))) | {'rows': len(rows), 'keys': len(keys), 'emotions': sorted(emotions)}


class Selector:
    def __init__(self):
        self.epoch = None
        self.scores = None

    def consider(self, scores, epoch):
        candidate = (scores['pcc'], -scores['l1'], -epoch)
        if not np.isfinite(candidate).all():
            raise ValueError('nonfinite selection metrics')
        previous = None if self.epoch is None else (self.scores['pcc'], -self.scores['l1'], -self.epoch)
        if previous is None or candidate > previous:
            self.epoch, self.scores = int(epoch), dict(scores)
            return True
        return False
