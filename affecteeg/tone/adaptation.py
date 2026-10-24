"""Train/validation-only fits with paired initialization and bounded search.

This module receives prepared feature arrays.
"""
from __future__ import annotations

import copy
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import adaptation_math as old
from . import interfaces as baseline
from .io import load_checkpoint, save_checkpoint

SCHEMA = "affecteeg-tone-adaptation-v1"


def nuisance_view(emotion, labels, view):
    z, stats = old.teacher_targets(emotion, labels)
    if not z.shape[1]:
        raise ValueError("teacher has no active training dimensions")
    result = {"teacher_statistics": stats, "view": view}
    if view != "original":
        if view not in ("train_only_pca32_no_whitening", "train_only_pca64_no_whitening"):
            raise ValueError(f"unknown teacher view: {view}")
        q = 32 if "pca32" in view else 64
        # Training population covariance; deterministic full eigendecomposition.
        center = z.mean(0, dtype=np.float64)
        centered = z.astype(np.float64) - center
        values, vectors = np.linalg.eigh(centered.T @ centered / len(z))
        keep = np.argsort(values)[::-1][:min(q, z.shape[1])]
        keep = keep[values[keep] > 1e-10]
        if not len(keep):
            raise ValueError("PCA nuisance is not estimable")
        components = vectors[:, keep]
        # Fix eigenvector sign for inspectable, reproducible projection records.
        pivots = np.argmax(np.abs(components), axis=0)
        components *= np.where(components[pivots, np.arange(len(keep))] < 0, -1, 1)
        z = (centered @ components).astype(np.float32)
        result.update(pca_mean=center, pca_components=components.astype(np.float32),
                      eigenvalues=values[keep], whitening=False)
    return z, result


def cov_loss(x, z, labels, mode):
    if mode == "global_centered":
        return old.covariance_penalty(x, z)
    if mode != "within_tone_centered_equal_tone_weight":
        raise ValueError(f"unknown covariance mode {mode}")
    losses = []
    for tone in range(4):
        mask = labels == tone
        if int(mask.sum()) < 2:
            raise ValueError("conditional covariance requires at least two trials per tone")
        losses.append(old.covariance_penalty(x[mask], z[mask]))
    return torch.stack(losses).mean()


def balanced_indices(labels, count, generator):
    if count % 4:
        raise ValueError("covariance/calibration batch must be divisible by four")
    groups = []
    for tone in range(4):
        rows = torch.as_tensor(np.flatnonzero(labels == tone))
        if len(rows) < count // 4:
            raise ValueError("insufficient training rows for balanced batch")
        groups.append(rows[torch.randperm(len(rows), generator=generator)[:count // 4]])
    return torch.cat(groups)


class FitModel(nn.Module):
    def __init__(self, contract, statistics, config, reference, arm):
        super().__init__()
        self.tail = baseline.build_tail(contract).float().eval().requires_grad_(False)
        self.head = baseline.build_head(contract).float()
        self.head.load_state_dict(reference["head_state"], strict=True)
        self.head.requires_grad_(config["head_lr"] > 0)
        self.adapter = None if arm == "identity" else old.RelativeAdapter(
            statistics, rank=config["rank"], scale=config["rho"])

    def forward(self, values, indices):
        x = values["target"][indices]
        clean = x if self.adapter is None else self.adapter(x)
        bypass = values.get("bypass")
        logits = self.head(baseline.tail_forward(self.tail, clean, None if bypass is None else bypass[indices]))
        return logits, clean


def make_model(contract, statistics, config, reference, arm, seed, device, model_factory=FitModel):
    old._seed(seed)
    return model_factory(contract, statistics, config, reference, arm).to(device)


def tensors(state, device):
    return {k: torch.as_tensor(state[k], dtype=torch.float32, device=device)
            for k in ("target", "bypass") if k in state}


@torch.no_grad()
def logits(model, values, batch_size=128):
    previous = model.head.training
    model.head.eval()
    out = [model(values, slice(i, i + batch_size))[0].cpu().numpy()
           for i in range(0, len(values["target"]), batch_size)]
    model.head.train(previous)
    return np.concatenate(out).astype(np.float32)


def calibrate(contract, train, labels, nuisance, statistics, config, reference, seed, device, model_factory=FitModel):
    if isinstance(config["covariance_weight"], (int, float)):
        return {"lambda": float(config["covariance_weight"]), "method": "fixed", "batches": []}
    model = make_model(contract, statistics, config, reference, "arra", seed, device, model_factory)
    model.head.eval()
    values = tensors(train, device)
    z = torch.as_tensor(nuisance, device=device)
    y = torch.as_tensor(labels, device=device)
    stats = {k: torch.as_tensor(v, device=device) for k, v in statistics.items()}
    generator = torch.Generator().manual_seed(seed + 7919)
    parameters = list(model.adapter.parameters())
    rows, ratios = [], []
    for _ in range(8):
        ce_idx = balanced_indices(labels, min(128, len(labels)), generator).to(device)
        cov_idx = balanced_indices(labels, min(config["covariance_batch_size"], len(labels)), generator).to(device)
        task, _ = model(values, ce_idx)
        ce = F.cross_entropy(task, y[ce_idx])
        clean = model.adapter(values["target"][cov_idx])
        covariance = cov_loss(old.standardized_view(clean, stats), z[cov_idx], y[cov_idx], config["covariance"])
        ce_norm = old._norm(torch.autograd.grad(ce, parameters, allow_unused=True))
        cov_norm = old._norm(torch.autograd.grad(covariance, parameters, allow_unused=True))
        ratio = cov_norm / ce_norm if min(ce_norm, cov_norm) > 1e-10 else None
        rows.append({"ce_gradient": ce_norm, "covariance_gradient": cov_norm, "ratio": ratio})
        if ratio is not None and math.isfinite(ratio):
            ratios.append(ratio)
    if len(ratios) < 4:
        return {"lambda": 10.0, "method": "near_zero_gradient_fixed_fallback", "batches": rows}
    median = float(np.median(ratios))
    raw = 0.1 / median
    return {"lambda": float(np.clip(raw, .1, 300.)), "unclipped_lambda": raw,
            "target_ratio": .1, "unweighted_ratio_median": median,
            "method": "training_gradient_calibration", "batches": rows}


@torch.no_grad()
def residual_metrics(model, values, stats, rho):
    if model.adapter is None:
        return {"normalized_residual_rms": 0., "relative_residual_norm": 0., "saturation_fraction": 0.}
    total, target_total, square, saturated, count = 0., 0., 0., 0, 0
    active = stats["active"].flatten()
    for start in range(0, len(values["target"]), 128):
        x = values["target"][start:start+128]
        delta = model.adapter(x) - x
        normalized = (delta / stats["safe_sigma"]).flatten(1)[:, active]
        total += float(delta.square().sum())
        target_total += float(x.square().sum())
        square += float(normalized.square().sum())
        saturated += int((normalized.abs() >= .95 * rho).sum())
        count += normalized.numel()
    return {"normalized_residual_rms": math.sqrt(square / max(count, 1)),
            "relative_residual_norm": math.sqrt(total / max(target_total, 1e-30)),
            "saturation_fraction": saturated / max(count, 1)}


def fit(*, contract, train, validation, train_labels, validation_labels, nuisance,
        statistics, config, reference, arm, calibration, seed, device="cpu",
        epochs=30, batch_size=128, checkpoint_path=None, expected=None, model_factory=FitModel):
    """Resume epoch checkpoints including AdamW, RNG and the current best state."""
    started = time.monotonic()
    device = torch.device(device)
    model = make_model(contract, statistics, config, reference, arm, seed, device, model_factory)
    tv, vv = tensors(train, device), tensors(validation, device)
    y = torch.as_tensor(train_labels, device=device)
    teacher = torch.as_tensor(nuisance, device=device)
    ref_logits = torch.as_tensor(reference["train_logits"], device=device)
    stats = {k: torch.as_tensor(v, device=device) for k, v in statistics.items()}
    groups = []
    if config["head_lr"] > 0:
        groups.append({"params": list(model.head.parameters()), "lr": config["head_lr"]})
    if model.adapter is not None:
        groups.append({"params": list(model.adapter.parameters()), "lr": config["adapter_lr"]})
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4) if groups else None
    parameters = [p for p in model.head.parameters() if p.requires_grad]
    nhead = len(parameters)
    parameters += [] if model.adapter is None else list(model.adapter.parameters())
    ce_generator = torch.Generator().manual_seed(seed)
    cov_generator = torch.Generator().manual_seed(seed + 65537)
    history, best, best_key, steps, start_epoch, previous_seconds = [], None, (-math.inf, -math.inf), 0, 1, 0.
    checkpoint_path = None if checkpoint_path is None else Path(checkpoint_path)

    def record(epoch, losses, gradients):
        nonlocal best, best_key
        validation_logits = logits(model, vv, batch_size)
        metrics = old._metrics(validation_labels, validation_logits)
        score = old._metric_key(metrics)
        diagnostics = residual_metrics(model, tv, stats, config["rho"])
        if score > best_key:
            best_key = score
            best = {"head_state": old._cpu_state(model.head),
                    "adapter_state": None if model.adapter is None else old._cpu_state(model.adapter),
                    "selected_epoch": epoch, "validation_metrics": metrics,
                    "validation_logits": validation_logits, "residual": diagnostics}
        history.append({"epoch": epoch, "optimizer_steps": steps, "validation_metrics": metrics,
                        "losses": losses, "gradients_first_batch": gradients, **diagnostics})

    initial_logits = logits(model, vv, batch_size)
    if not np.allclose(initial_logits, reference["validation_logits"], atol=2e-5, rtol=2e-5):
        raise RuntimeError("zero adapter does not reproduce reference A validation logits")
    record(0, {}, {})
    if checkpoint_path is not None and checkpoint_path.exists():
        saved = load_checkpoint(checkpoint_path)
        if not old._equal(saved["expected"], expected):
            raise RuntimeError("in-progress checkpoint configuration/source mismatch")
        model.load_state_dict(saved["model_state"])
        if optimizer is not None:
            optimizer.load_state_dict(saved["optimizer_state"])
        history, best, best_key = saved["history"], saved["best"], tuple(saved["best_key"])
        steps, start_epoch, previous_seconds = saved["steps"], saved["epoch"] + 1, saved["elapsed_seconds"]
        ce_generator.set_state(saved["ce_rng"])
        cov_generator.set_state(saved["cov_rng"])
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(saved["cuda_rng"], device)
    for epoch in range(start_epoch, epochs + 1):
        model.head.train()
        model.tail.eval()
        permutation = torch.randperm(len(train_labels), generator=ce_generator)
        sums, gradient_report = {}, {}
        for batch, start in enumerate(range(0, len(train_labels), batch_size)):
            idx = permutation[start:start+batch_size].to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            pred, clean = model(tv, idx)
            ce = F.cross_entropy(pred, y[idx])
            terms = {"ce": ce, "native_regularizer": old._head_regularizer(model.head, ce)}
            terms["kd"] = config["kd_weight"] * 4 * F.kl_div(
                F.log_softmax(pred / 2, dim=1), F.softmax(ref_logits[idx] / 2, dim=1), reduction="batchmean")
            if model.adapter is not None:
                normalized = ((clean - tv["target"][idx]) / stats["safe_sigma"]).flatten(1)[:, stats["active"].flatten()]
                terms["preservation"] = .01 * normalized.square().mean()
            if arm in ("arra", "within_tone_shuffle"):
                if config["covariance_batch_size"] == 128:
                    cidx, cov_clean = idx, clean
                else:
                    cidx = balanced_indices(train_labels, min(config["covariance_batch_size"], len(train_labels)), cov_generator).to(device)
                    cov_clean = model.adapter(tv["target"][cidx])
                terms["covariance"] = calibration["lambda"] * cov_loss(
                    old.standardized_view(cov_clean, stats), teacher[cidx], y[cidx], config["covariance"])
            loss = sum(terms.values())
            if batch == 0:
                for name, term in terms.items():
                    if term.requires_grad:
                        grad = torch.autograd.grad(term, parameters, retain_graph=True, allow_unused=True)
                        gradient_report[name] = {"head": old._norm(grad[:nhead]), "adapter": old._norm(grad[nhead:])}
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite training loss")
            if optimizer is not None:
                loss.backward()
                if not all(p.grad is None or torch.isfinite(p.grad).all() for p in parameters):
                    raise FloatingPointError("nonfinite gradient")
                optimizer.step()
                steps += 1
            for key, value in terms.items():
                sums[key] = sums.get(key, 0.) + float(value.detach()) * len(idx)
        record(epoch, {k: v / len(train_labels) for k, v in sums.items()}, gradient_report)
        if checkpoint_path is not None:
            save_checkpoint(checkpoint_path, {"expected": expected, "epoch": epoch,
                "model_state": old._cpu_state(model), "optimizer_state": None if optimizer is None else optimizer.state_dict(),
                "history": history, "best": best, "best_key": best_key, "steps": steps,
                "ce_rng": ce_generator.get_state(), "cov_rng": cov_generator.get_state(),
                "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                "elapsed_seconds": previous_seconds + time.monotonic() - started})
    model.head.load_state_dict(best["head_state"])
    if model.adapter is not None:
        model.adapter.load_state_dict(best["adapter_state"])
    return {"schema": SCHEMA, **best, "history": history, "optimizer_steps": steps,
            "initial_validation_logits": initial_logits, "initial_head_state": copy.deepcopy(reference["head_state"]),
            "train_logits": logits(model, tv, batch_size), "normalization": copy.deepcopy(statistics),
            "config": copy.deepcopy(config), "arm": arm, "calibration": copy.deepcopy(calibration),
            "elapsed_seconds": previous_seconds + time.monotonic() - started,
            "precision": "float32", "outer_test_accessed": False,
            "runtime_environment": {"torch": str(torch.__version__), "numpy": str(np.__version__), "device": str(device)}}


def predict(artifact, contract, state, device="cpu", model_factory=FitModel):
    model = model_factory(contract, artifact["normalization"], artifact["config"], artifact, artifact["arm"]).to(device)
    if model.adapter is not None:
        model.adapter.load_state_dict(artifact["adapter_state"], strict=True)
    return logits(model, tensors(state, device))
