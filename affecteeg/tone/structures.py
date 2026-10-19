"""Validation-triggered alternative interfaces from the supplementary protocol."""
from __future__ import annotations
from collections import OrderedDict
import numpy as np
import torch
from torch import nn
from . import adaptation_math as old, interfaces as baseline, adaptation as a

class GroupAdapter(nn.Module):
    """Deformer output is 960 flattened features followed by four 64-d blocks."""
    def __init__(self, statistics, rank, scale):
        super().__init__()
        if tuple(statistics["mean"].shape) != (1216,):
            raise ValueError("registered Deformer multi-scale layout must be 1216")
        self.widths = (960, 64, 64, 64, 64)
        modules, start = [], 0
        for width in self.widths:
            stats = {k: v[start:start+width] for k, v in statistics.items()}
            modules.append(old.RelativeAdapter(stats, rank=min(rank, width // 2), scale=scale))
            start += width
        self.blocks = nn.ModuleList(modules)

    def forward(self, value):
        return torch.cat([module(block) for module, block in zip(self.blocks, value.split(self.widths, -1))], -1)

class StructuralModel(a.FitModel):
    def __init__(self, contract, statistics, config, reference, arm):
        super().__init__(contract, statistics, config, reference, arm)
        self.mode = config["feature_mode"]
        if self.mode == "conformer256":
            # Sequential slicing retains the original classifier keys 3..6.
            self.tail = nn.Sequential(OrderedDict([
                ("3", nn.Dropout(.5)), ("4", nn.Linear(256, 32)),
                ("5", nn.ELU()), ("6", nn.Dropout(.5))]))
            self.tail.load_state_dict(contract["structural_tail_state"], strict=True)
            self.tail.requires_grad_(False).eval()
        if self.mode == "deformer_multiscale" and arm != "identity":
            self.adapter = GroupAdapter(statistics, config["rank"], config["rho"])

    def forward(self, values, indices):
        x = values["target"][indices]
        clean = x if self.adapter is None else self.adapter(x)
        if self.mode == "cat_dual_pool":
            features = baseline.tail_forward(self.tail, clean[:, :256], clean[:, 256:])
        elif self.mode == "reve_after_norm":
            features = clean
        elif self.mode == "conformer256":
            features = self.tail(clean)
        elif self.mode == "deformer_multiscale":
            features = baseline.tail_forward(self.tail, clean, None)
        else:
            raise ValueError(self.mode)
        return self.head(features), clean

def cached_transform(state, contract, mode):
    x = np.asarray(state["target"], dtype=np.float32)
    if mode == "cat_dual_pool":
        return {"target": np.concatenate((x, state["bypass"]), 1)}
    if mode == "reve_after_norm":
        tail = baseline.build_tail(contract).cpu().eval()
        return {"target": np.concatenate([tail(torch.tensor(x[i:i+128]), None).numpy() for i in range(0, len(x), 128)])}
    if mode == "deformer_multiscale":
        return {"target": x}
    raise ValueError("this mode requires earlier-layer feature extraction")
