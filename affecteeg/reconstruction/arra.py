"""Exact bounded ARRA and conditional covariance used by the full model."""
from __future__ import annotations
from contextlib import contextmanager
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
CONFIG = {"pool_bins": 8, "auxiliary_groups": 4, "auxiliary_sources_per_group": 8}
POOL_BINS = 8
POOL_SCHEMA = "affecteeg_pool_train_statistics_v1"

def deterministic_adaptive_pool(sequence, bins):
    """Adaptive-average bin boundaries with a deterministic sliced-mean backward."""
    length = sequence.shape[-1]
    return torch.stack([sequence[..., (index * length) // bins:((index + 1) * length + bins - 1) // bins].mean(-1)
                        for index in range(bins)], dim=-1)

def deterministic_double_linear(sequence):
    """Exactly doubled align_corners=False linear interpolation, without atomics."""
    left = torch.cat((sequence[..., :1], sequence[..., :-1]), dim=-1)
    right = torch.cat((sequence[..., 1:], sequence[..., -1:]), dim=-1)
    return torch.stack((.75 * sequence + .25 * left, .75 * sequence + .25 * right), dim=-1).flatten(-2)

class DeterministicSSM2MelCore(nn.Module):
    """Keep all original SSM2Mel parameters and change only its doubling kernel."""
    def __init__(self, original):
        super().__init__()
        for name in ("stem", "down", "forward_state", "backward_state", "skip_mix"):
            setattr(self, name, getattr(original, name))
        self.out_dim = original.out_dim

    def forward(self, x):
        skip = self.stem(x)
        downsampled = F.gelu(self.down(skip.transpose(1, 2))).transpose(1, 2)
        forward, backward = downsampled, torch.flip(downsampled, dims=[1])
        for forward_block, backward_block in zip(self.forward_state, self.backward_state):
            forward, backward = forward_block(forward), backward_block(backward)
        hidden = .5 * (forward + torch.flip(backward, dims=[1]))
        if skip.size(1) != 2 * hidden.size(1):
            raise ValueError("the registered 800-sample SSM2Mel path requires exact doubling")
        up = deterministic_double_linear(hidden.transpose(1, 2)).transpose(1, 2)
        return self.skip_mix(torch.cat((skip, up), dim=-1))

@contextmanager
def _auxiliary_evaluation(module, *, gradients):
    """Keep eval semantics and isolate auxiliary RNG without disabling cuDNN.

    A cuDNN RNN needs its training reserve for backward, even when its dropout
    is zero. Only RNNBase instances enter training mode in the gradient path;
    all BatchNorm and external dropout modules remain in evaluation mode.
    """
    states = [(m, m.training) for m in module.modules()]
    recurrent = [(m, m.dropout) for m, _ in states if isinstance(m, nn.RNNBase)]
    devices = sorted({tensor.device.index for tensor in (*module.parameters(), *module.buffers())
                      if tensor.device.type == "cuda"})
    python_state, numpy_state = random.getstate(), np.random.get_state()
    module.eval()
    try:
        for child, _ in recurrent:
            child.dropout = 0.0
            child.training = bool(gradients)
        # Native eval RNNs used IEEE FP32. Keep that precision in the faster
        # auxiliary cuDNN path; TF32 convolution otherwise changes memory by
        # about 3e-4 on the registered NeuroTalk inputs.
        backend = torch.backends.cudnn.flags(
            enabled=torch.backends.cudnn.enabled, benchmark=torch.backends.cudnn.benchmark,
            benchmark_limit=torch.backends.cudnn.benchmark_limit,
            deterministic=torch.backends.cudnn.deterministic, allow_tf32=False)
        with torch.random.fork_rng(devices=devices), torch.set_grad_enabled(gradients), backend:
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        for child, dropout in recurrent:
            child.dropout = dropout
        for child, state in states:
            child.training = state

@contextmanager
def evaluation_without_grad(module):
    with _auxiliary_evaluation(module, gradients=False):
        yield

class BoundedContentAdapter(nn.Module):
    def __init__(self, mean, sigma, rank=16, rho=0.1):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("sigma", torch.as_tensor(sigma, dtype=torch.float32))
        self.register_buffer("active", self.sigma > 1e-6)
        self.register_buffer("safe_sigma", torch.where(self.active, self.sigma, 1.0))
        dim = self.mean.shape[-1]
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, rank)
        self.up = nn.Linear(rank, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.rho = float(rho)

    def forward(self, memory):
        standard = (memory - self.mean) / self.safe_sigma * self.active
        correction = self.up(F.gelu(self.down(self.norm(standard))))
        return memory - self.rho * self.sigma * self.active * torch.tanh(correction)

    def view(self, memory):
        standard = (memory - self.mean) / self.safe_sigma * self.active
        return deterministic_adaptive_pool(standard.transpose(1, 2), CONFIG["pool_bins"]).flatten(1)

    def preservation(self, before, after):
        difference = (after - before) / self.safe_sigma * self.active
        return difference.square().sum() / (before.shape[0] * self.active.sum().clamp_min(1))

def grouped_covariance(content, teacher, group_ids, *, group_size=None):
    """Average squared within-group cross-covariance.

    ``group_size`` uses batched matmul for the sampler's contiguous 4x8 layout.
    Its labels must remain on the CPU, so layout validation does not synchronize
    the GPU. The generic path retains support for unordered, unequal groups.
    """
    if content.ndim != 2 or teacher.ndim != 2 or len(content) != len(teacher):
        raise ValueError("covariance inputs must be aligned two-dimensional matrices")
    if len(group_ids) != len(content):
        raise ValueError("covariance group labels do not match the sample count")
    if group_size is not None:
        size = int(group_size)
        if size < 2 or not len(content) or len(content) % size:
            raise ValueError("fixed covariance groups need equal blocks of at least two sources")
        labels = torch.as_tensor(group_ids)
        if labels.device.type != "cpu":
            raise ValueError("keep fixed-layout auxiliary group labels on CPU")
        labels = labels.reshape(-1, size)
        if not torch.all(labels == labels[:, :1]) or len(torch.unique(labels[:, 0])) != len(labels):
            raise ValueError("fixed covariance requires distinct contiguous group blocks")
        x, z = content.reshape(-1, size, content.shape[1]), teacher.detach().reshape(-1, size, teacher.shape[1])
        x, z = x - x.mean(1, keepdim=True), z - z.mean(1, keepdim=True)
        return (torch.bmm(x.transpose(1, 2), z) / size).square().mean()
    group_ids = torch.as_tensor(group_ids, device=content.device)
    losses = []
    for group in torch.unique(group_ids):
        selected = group_ids == group
        if int(selected.sum()) < 2:
            raise ValueError("conditional covariance needs at least two distinct sources")
        x, z = content[selected], teacher[selected].detach()
        x, z = x - x.mean(0), z - z.mean(0)
        losses.append((x.T @ z / len(x)).square().mean())
    if not losses:
        raise ValueError("empty auxiliary covariance batch")
    return torch.stack(losses).mean()

def raw_pool(memory):
    if memory.ndim != 3 or tuple(memory.shape[1:]) != (800, 256):
        raise ValueError('registered strong APC memory must have shape [B,800,256]')
    return deterministic_adaptive_pool(memory.transpose(1, 2), POOL_BINS).flatten(1)

class CommonPoolView:
    def register_pool(self, stats):
        if stats.get('schema') != POOL_SCHEMA or stats.get('statistics_split') != 'train':
            raise ValueError('unregistered or non-training pool statistics')
        mean = torch.as_tensor(stats['mean'], dtype=torch.float32).clone()
        sigma = torch.as_tensor(stats['sigma'], dtype=torch.float32).clone()
        active = sigma > 1e-6
        if mean.shape != (2048,) or sigma.shape != (2048,) or not torch.isfinite(mean).all() or not torch.isfinite(sigma).all():
            raise ValueError('invalid pooled source statistics')
        if bool((sigma < 0).any()) or not torch.equal(active, torch.as_tensor(stats['active']).bool()):
            raise ValueError('pool active mask differs from registered variance rule')
        self.register_buffer('pool_mean', mean)
        self.register_buffer('pool_sigma', sigma)
        self.register_buffer('pool_active', active)
        self.register_buffer('pool_safe_sigma', torch.where(active, sigma, 1.))

    def view(self, memory):
        return (raw_pool(memory) - self.pool_mean) / self.pool_safe_sigma * self.pool_active

class StrongBoundedAdapter(CommonPoolView, BoundedContentAdapter):
    def __init__(self, feature_statistics, pool_statistics):
        super().__init__(feature_statistics['mean'], feature_statistics['sigma'], rank=16, rho=.1)
        self.register_pool(pool_statistics)

    def set_phase_epoch(self, epoch):
        pass

def covariance_correction(task, covariance):
    """Return gCprime-gC and inspectable scalar geometry; do not mutate inputs.

    Reductions use float64 to avoid a spurious zero norm from FP32 squaring.
    The correction itself retains each parameter's FP32 dtype.
    """
    if len(task) != len(covariance) or not task:
        raise ValueError('aligned, nonempty adapter gradient lists are required')
    if any(a.shape != b.shape for a,b in zip(task,covariance)):
        raise ValueError('adapter gradient shapes differ')
    task_sq = sum(value.detach().double().square().sum() for value in task)
    cov_sq = sum(value.detach().double().square().sum() for value in covariance)
    dot = sum((a.detach().double()*b.detach().double()).sum() for a,b in zip(task,covariance))
    if not bool(torch.isfinite(torch.stack((task_sq,cov_sq,dot))).all()):
        raise ValueError('nonfinite adapter gradient geometry')
    project = bool((dot < 0) & (task_sq > 0))
    correction = None
    removed_sq = torch.zeros_like(dot)
    actual_after_dot = dot
    if project:
        factor = -dot/task_sq
        correction = tuple(a.detach()*factor.to(a.dtype) for a in task)
        removed_sq = sum(value.double().square().sum() for value in correction)
        actual_after_dot = sum((a.detach().double()*(b.detach()+c).double()).sum()
                               for a,b,c in zip(task,covariance,correction))
    task_value,cov_value,dot_value,removed_value,after_value = torch.stack(
        (task_sq,cov_sq,dot,removed_sq,actual_after_dot)).cpu().tolist()
    cosine = dot_value/(task_value*cov_value)**.5 if task_value>0 and cov_value>0 else None
    stats = dict(task_squared_norm=task_value,covariance_squared_norm=cov_value,
        raw_dot=dot_value,raw_cosine=cosine,negative_cosine=bool(cosine is not None and cosine<0),
        projection_applied=project,removed_squared_norm=removed_value,
        removed_covariance_energy_fraction=removed_value/cov_value if cov_value>0 else None,
        projected_raw_dot=after_value)
    return correction,stats

def fit_targets(embeddings, groups, splits, dimensions=32):
    """Only training rows determine conditional means, scales, and PCA basis."""
    embeddings = np.asarray(embeddings, np.float64)
    groups, splits = np.asarray(groups), np.asarray(splits)
    train = splits == "train"
    names = np.unique(groups)
    means = []
    residual = embeddings.copy()
    for group in names:
        support = (groups == group) & train
        if support.sum() < 8:
            raise ValueError(f"training group {group} has fewer than eight content sources")
        mean = embeddings[support].mean(0)
        means.append(mean)
        residual[groups == group] -= mean
    scale = residual[train].std(0)
    active = scale > 1e-6
    standardized = residual[:, active] / scale[active]
    center = standardized[train].mean(0)
    centered = standardized - center
    eigenvalues, eigenvectors = np.linalg.eigh(centered[train].T @ centered[train] / train.sum())
    order = np.argsort(eigenvalues)[::-1][:dimensions]
    if len(order) != dimensions or eigenvalues[order[-1]] <= 1e-10:
        raise ValueError("training teacher has fewer than 32 estimable PCA dimensions")
    components = eigenvectors[:, order]
    pivots = np.argmax(abs(components), axis=0)
    signs = np.sign(components[pivots, np.arange(dimensions)])
    components *= np.where(signs == 0, 1, signs)
    targets = np.asarray(centered @ components, np.float32)
    stats = {"group_names": names, "group_means": np.asarray(means, np.float32),
             "residual_scale": scale.astype(np.float32), "active": active,
             "pca_mean": center.astype(np.float32), "pca_components": components.astype(np.float32),
             "pca_eigenvalues": eigenvalues[order].astype(np.float32),
             "training_rows": np.flatnonzero(train)}
    return targets, stats

class AuxiliarySampler:
    """A separate CPU generator samples four groups and eight distinct sources each."""
    def __init__(self, teacher, seed):
        self.generator = torch.Generator(device="cpu").manual_seed(int(seed) + 900_001)
        train = teacher["splits"] == "train"
        names = np.unique(teacher["groups"][train])
        self.groups = [np.flatnonzero(train & (teacher["groups"] == group)) for group in names]
        if len(self.groups) < CONFIG["auxiliary_groups"] or any(len(rows) < CONFIG["auxiliary_sources_per_group"] for rows in self.groups):
            raise ValueError("source pool cannot support the fixed 4x8 auxiliary design")

    def sample(self):
        chosen = torch.randperm(len(self.groups), generator=self.generator)[:CONFIG["auxiliary_groups"]].tolist()
        rows, labels = [], []
        for group in chosen:
            pool = self.groups[group]
            selected = torch.randperm(len(pool), generator=self.generator)[:CONFIG["auxiliary_sources_per_group"]].numpy()
            rows.extend(pool[selected].tolist())
            labels.extend([group] * len(selected))
        return np.asarray(rows, np.int64), np.asarray(labels, np.int64)

    def state_dict(self):
        return {"generator_state": self.generator.get_state()}

    def load_state_dict(self, state):
        self.generator.set_state(state["generator_state"])
