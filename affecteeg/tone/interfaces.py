"""Exact native feature boundaries, frozen downstream layers and cloned heads."""
from __future__ import annotations
import copy, math
from typing import Any, Mapping
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .protocol import BACKBONES, FEATURE_DIMS

class StructuredATCHead(nn.Module):
    def __init__(self, classifiers: nn.ModuleList) -> None:
        super().__init__()
        self.classifiers = copy.deepcopy(classifiers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or tuple(features.shape[1:]) != (5, 32):
            raise ValueError(f"ATC feature shape drift: {tuple(features.shape)}")
        return torch.stack(
            [layer(features[:, index]) for index, layer in enumerate(self.classifiers)],
            dim=0,
        ).mean(dim=0)

    def regularization_loss(self) -> torch.Tensor:
        return 0.5 * sum(layer.weight.square().sum() for layer in self.classifiers)


class CATFusionTail(nn.Module):
    def __init__(self, linear: nn.Linear, normalization: nn.BatchNorm1d, dropout: nn.Dropout):
        super().__init__()
        self.linear = copy.deepcopy(linear)
        self.normalization = copy.deepcopy(normalization)
        self.dropout = copy.deepcopy(dropout)

    def forward(self, target: torch.Tensor, bypass: torch.Tensor | None) -> torch.Tensor:
        if bypass is None:
            raise ValueError("CAT-Net requires the frozen EMG bypass")
        value = F.relu(self.linear(torch.cat([target, bypass], dim=1)))
        return self.dropout(self.normalization(value))


class ExactRMSNorm(nn.Module):
    def __init__(self, dimension: int, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dimension))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        normalized = value.float() * torch.rsqrt(
            value.float().pow(2).mean(-1, keepdim=True) + self.eps
        )
        return normalized.to(value.dtype) * self.weight


class REVETail(nn.Module):
    def __init__(self, norm: nn.Module, dropout: nn.Dropout) -> None:
        super().__init__()
        dimension = int(norm.weight.numel())
        self.norm = ExactRMSNorm(dimension, float(getattr(norm, "eps", 1.0e-6)))
        self.norm.load_state_dict(norm.state_dict(), strict=True)
        self.dropout = nn.Dropout(float(dropout.p))

    def forward(self, target: torch.Tensor, bypass: torch.Tensor | None) -> torch.Tensor:
        del bypass
        return self.dropout(self.norm(target))


class IdentityTail(nn.Module):
    def forward(self, target: torch.Tensor, bypass: torch.Tensor | None) -> torch.Tensor:
        del bypass
        return target


class FeatureExtractor(nn.Module):
    """Frozen exact feature boundary for each registered backbone."""

    def __init__(self, backbone: str, model: nn.Module) -> None:
        super().__init__()
        if backbone not in BACKBONES: raise ValueError(backbone)
        self.backbone_id = backbone
        self.model = model
        self.model.eval()

    @property
    def feature_dim(self) -> int:
        return FEATURE_DIMS[self.backbone_id]

    def extract(self, eeg: torch.Tensor, emg: torch.Tensor) -> dict[str, torch.Tensor]:
        name, model = self.backbone_id, self.model
        if name == "cat_net":
            if model.eeg_input_indices.numel():
                eeg = eeg.index_select(-1, model.eeg_input_indices)
            ev = eeg.permute(0, 2, 1)
            ev = F.relu(model.eeg_bn1(model.eeg_conv1(ev)))
            ev = F.relu(model.eeg_bn2(model.eeg_conv2(ev)))
            ev = model.eeg_channel_attn(model.eeg_pool(ev)).permute(0, 2, 1)
            ev, _ = model.eeg_lstm1(ev)
            ev, _ = model.eeg_lstm2(ev)
            ev = model.eeg_dropout(model.eeg_ln(ev))
            mv = emg.permute(0, 2, 1)
            mv = F.relu(model.emg_bn1(model.emg_conv1(mv)))
            mv = F.relu(model.emg_bn2(model.emg_conv2(mv)))
            mv = model.emg_channel_attn(model.emg_pool(mv)).permute(0, 2, 1)
            mv, _ = model.emg_lstm1(mv)
            mv, _ = model.emg_lstm2(mv)
            mv = model.emg_dropout(model.emg_ln(mv))
            en, mn = model.eeg_pre_norm(ev), model.emg_pre_norm(mv)
            ev, _ = model.cross_attn_eeg(en, mn, mn)
            mv, _ = model.cross_attn_emg(mn, en, en)
            ev = model.eeg_post_norm2(model.eeg_post_norm1(ev))
            mv = model.emg_post_norm2(model.emg_post_norm1(mv))
            return {
                "target": torch.cat([ev.mean(1), ev.max(1).values], dim=1),
                "bypass": torch.cat([mv.mean(1), mv.max(1).values], dim=1),
            }
        raw = model.raw_channels(eeg, model.eeg_channels)
        if name in {"eegnet", "atcnet", "eeg_conformer"}:
            value = raw.permute(0, 2, 1).unsqueeze(1)
        if name == "eegnet":
            value = model.backbone.block2(model.backbone.block1(value))
            return {"target": value.flatten(1)}
        if name == "atcnet":
            sequence = model.backbone.conv_block(value).squeeze(2).transpose(1, 2)
            windows = []
            for index, (start, stop) in enumerate(model.backbone.window_bounds):
                attended = model.backbone.attention[index](sequence[:, start:stop])
                windows.append(model.backbone.tcns[index](attended)[:, -1])
            return {"target": torch.stack(windows, dim=1)}
        if name == "eeg_conformer":
            tokens = model.backbone.encoder(model.backbone.embedding(value))
            return {"target": model.backbone.classifier[:-1](tokens)}
        if name == "rgnn":
            nodes = raw[:, :, model.channel_reorder].transpose(1, 2)
            nodes = F.relu(model.graph_conv(nodes, model.adjacency_matrix()))
            return {"target": F.dropout(nodes.sum(1), model.dropout, False)}
        if name == "cbramod":
            patches, mask = model.adapt_eeg(eeg)
            return {"target": model.backbone(patches, mask=mask).mean((1, 2))}
        if name == "st_eegformer":
            signal = model.adapt_eeg(eeg)
            indices = model.channel_indices.unsqueeze(0).expand(signal.shape[0], -1)
            return {"target": model.backbone.forward_features(signal, indices)}
        if name == "eeg_deformer":
            base = model.backbone
            value = base.cnn_encoder(raw.transpose(1, 2).unsqueeze(1))
            value = base.to_patch_embedding(value)
            value = base.transformer(value + base.pos_embedding)
            return {"target": value}
        if name == "eegmamba":
            return {"target": model.backbone(model.adapt_eeg(eeg)).mean((1, 2))}
        if name == "reve":
            signal = model.adapt_eeg(eeg)
            positions = model.electrode_positions.unsqueeze(0).expand(signal.shape[0], -1, -1)
            tokens = model.backbone(signal, positions).flatten(1, 2)
            query = model.head.cls_query_token.expand(tokens.shape[0], -1, -1)
            weights = torch.softmax(
                torch.matmul(query, tokens.transpose(-1, -2))
                / math.sqrt(model.head.embed_dim),
                dim=-1,
            )
            return {"target": torch.matmul(weights, tokens).squeeze(1)}
        raise KeyError(name)

    def projection_view(self, target: torch.Tensor) -> torch.Tensor:
        return target.mean(1) if self.backbone_id == "atcnet" else target


def _linear_spec(layer: nn.Linear) -> dict[str, Any]:
    return {
        "kind": "linear",
        "in_features": int(layer.in_features),
        "out_features": int(layer.out_features),
        "bias": layer.bias is not None,
        "state_dict": {key: value.detach().cpu() for key, value in layer.state_dict().items()},
    }


def contract_from_model(backbone: str, model: nn.Module) -> dict[str, Any]:
    if backbone == "cat_net":
        tail = CATFusionTail(model.fused_fc, model.fused_bn, model.fused_dropout)
        head = _linear_spec(model.class_head)
        tail_spec = {
            "kind": "cat_fusion",
            "in_features": int(model.fused_fc.in_features),
            "out_features": int(model.fused_fc.out_features),
            "bias": model.fused_fc.bias is not None,
            "dropout": float(model.fused_dropout.p),
            "state_dict": {key: value.detach().cpu() for key, value in tail.state_dict().items()},
        }
    elif backbone == "atcnet":
        tail_spec = {"kind": "identity", "state_dict": {}}
        head_module = StructuredATCHead(model.backbone.classifiers)
        head = {
            "kind": "atc_five_head",
            "in_features": 32,
            "out_features": int(model.backbone.classifiers[0].out_features),
            "bias": model.backbone.classifiers[0].bias is not None,
            "state_dict": {key: value.detach().cpu() for key, value in head_module.state_dict().items()},
        }
    else:
        tail_spec = {"kind": "identity", "state_dict": {}}
        if backbone == "eegnet":
            layer = model.backbone.lin
        elif backbone == "eeg_conformer":
            layer = model.backbone.classifier[-1]
        elif backbone == "rgnn":
            layer = model.classifier
        elif backbone in {"cbramod", "eegmamba"}:
            layer = model.classifier
        elif backbone == "st_eegformer":
            layer = model.backbone.head
        elif backbone == "eeg_deformer":
            layer = model.backbone.mlp_head[-1]
        elif backbone == "reve":
            norm = model.head.linear_head[0]
            dropout = model.head.linear_head[1]
            tail = REVETail(norm, dropout)
            tail_spec = {
                "kind": "reve_norm",
                "dimension": int(norm.weight.numel()),
                "eps": float(getattr(norm, "eps", 1.0e-6)),
                "dropout": float(dropout.p),
                "state_dict": {key: value.detach().cpu() for key, value in tail.state_dict().items()},
            }
            layer = model.head.linear_head[-1]
        else:
            raise KeyError(backbone)
        head = _linear_spec(layer)
    contract = {
        "schema_version": "part1-ten-backbone-feature-contract-v1",
        "backbone": backbone,
        "feature_dim": FEATURE_DIMS[backbone],
        "target_layout": "five_windows" if backbone == "atcnet" else "matrix",
        "projection_view": "window_mean" if backbone == "atcnet" else "identity",
        "tail": tail_spec,
        "head": head,
    }
    if backbone == "rgnn":
        contract["graph_metadata"] = dict(model.graph_metadata)
    return contract


def build_tail(contract: Mapping[str, Any]) -> nn.Module:
    spec = dict(contract["tail"])
    kind = spec["kind"]
    if kind == "identity":
        return IdentityTail()
    if kind == "cat_fusion":
        linear = nn.Linear(spec["in_features"], spec["out_features"], bias=spec["bias"])
        normalization = nn.BatchNorm1d(spec["out_features"])
        module = CATFusionTail(linear, normalization, nn.Dropout(spec["dropout"]))
    elif kind == "reve_norm":
        norm = ExactRMSNorm(spec["dimension"], spec["eps"])
        module = REVETail(norm, nn.Dropout(spec["dropout"]))
    else:
        raise ValueError(f"unknown tail kind: {kind}")
    module.load_state_dict(spec["state_dict"], strict=True)
    module.requires_grad_(False)
    module.eval()
    return module


def build_head(contract: Mapping[str, Any]) -> nn.Module:
    spec = dict(contract["head"])
    if spec["kind"] == "linear":
        module: nn.Module = nn.Linear(
            spec["in_features"], spec["out_features"], bias=spec["bias"]
        )
    elif spec["kind"] == "atc_five_head":
        layers = nn.ModuleList(
            [
                nn.Linear(spec["in_features"], spec["out_features"], bias=spec["bias"])
                for _ in range(5)
            ]
        )
        module = StructuredATCHead(layers)
    else:
        raise ValueError(f"unknown head kind: {spec['kind']}")
    module.load_state_dict(spec["state_dict"], strict=True)
    module.requires_grad_(True)
    return module


def tail_forward(
    tail: nn.Module, target: torch.Tensor, bypass: torch.Tensor | None
) -> torch.Tensor:
    tail.eval()
    return tail(target, bypass)


def projection_view_numpy(contract: Mapping[str, Any], target: np.ndarray) -> np.ndarray:
    value = np.asarray(target, dtype=np.float32)
    return value.mean(1) if contract["projection_view"] == "window_mean" else value


def assert_identity_parity(
    backbone: str,
    model: nn.Module,
    eeg: torch.Tensor,
    emg: torch.Tensor,
    *,
    atol: float = 2.0e-5,
    rtol: float = 2.0e-5,
) -> None:
    model.eval()
    extractor = FeatureExtractor(backbone, model)
    contract = contract_from_model(backbone, model)
    tail = build_tail(contract).to(eeg.device)
    head = build_head(contract).to(eeg.device).eval()
    with torch.no_grad():
        expected = model(eeg, emg)[0]
        state = extractor.extract(eeg, emg)
        observed = head(tail_forward(tail, state["target"], state.get("bypass")))
    if not torch.allclose(expected, observed, atol=atol, rtol=rtol):
        error = float((expected - observed).abs().max().item())
        raise RuntimeError(f"{backbone} identity reconstruction drift: max_abs={error:.6g}")

