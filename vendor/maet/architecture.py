"""MAET backbone (SEED-VII official architecture, timm-free vendored copy).

Reference: https://github.com/935963004/MAET
Paper: Jiang et al., IEEE TAFFC 2025 (SEED-VII).
"""

from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


def _drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = x.new_empty(shape).bernoulli_(keep).div(keep)
    return x * mask


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _drop_path(x, self.drop_prob, self.training)


def _trunc_normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0) -> None:
    nn.init.trunc_normal_(tensor, mean=mean, std=std, a=-2 * std, b=2 * std)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer=nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(dim)) if qkv_bias else None
        self.v_bias = nn.Parameter(torch.zeros(dim)) if qkv_bias else None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        relative_position_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, n, c = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias)
            )
        qkv = F.linear(x, self.qkv.weight, qkv_bias)
        qkv = qkv.reshape(b, n, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q.float() @ k.float().transpose(-2, -1)) * self.scale
        if relative_position_bias is not None:
            attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))
        attn = attn.softmax(dim=-1).type_as(x)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        return self.proj_drop(x)


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        with_mixffn: bool = False,
        layer_scale_init_values: float = 0.1,
        max_text_len: int = 40,
        prompt: bool = False,
        prompt_len: int = 2,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2_text = norm_layer(dim)
        self.norm2_imag = norm_layer(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp_text = Mlp(dim, mlp_hidden, act_layer=act_layer, drop=drop)
        self.mlp_imag = Mlp(dim, mlp_hidden, act_layer=act_layer, drop=drop)
        self.mlp_vl = Mlp(dim, mlp_hidden, act_layer=act_layer, drop=drop) if with_mixffn else None
        self.norm2_vl = norm_layer(dim) if with_mixffn else None
        self.gamma_1 = (
            nn.Parameter(layer_scale_init_values * torch.ones(dim), requires_grad=True)
            if layer_scale_init_values is not None
            else 1.0
        )
        self.gamma_2 = (
            nn.Parameter(layer_scale_init_values * torch.ones(dim), requires_grad=True)
            if layer_scale_init_values is not None
            else 1.0
        )
        self.max_text_len = max_text_len
        self.prompt_len = prompt_len
        self.eeg_prompt = nn.Parameter(torch.zeros(1, prompt_len, dim)) if prompt else None
        self.eye_prompt = nn.Parameter(torch.zeros(1, prompt_len, dim)) if prompt else None
        if self.eeg_prompt is not None:
            _trunc_normal_(self.eeg_prompt, std=0.02)
        if self.eye_prompt is not None:
            _trunc_normal_(self.eye_prompt, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        modality_type: str | None = None,
        relative_position_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if modality_type == "eeg" and self.eeg_prompt is not None:
            x = torch.cat((x, self.eeg_prompt.expand(x.size(0), -1, -1)), dim=1)
        elif modality_type == "eye" and self.eye_prompt is not None:
            x = torch.cat((x, self.eye_prompt.expand(x.size(0), -1, -1)), dim=1)

        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), mask, relative_position_bias))

        if modality_type == "eye":
            x = x + self.drop_path(self.gamma_2 * self.mlp_imag(self.norm2_imag(x)))
        elif modality_type == "eeg":
            x = x + self.drop_path(self.gamma_2 * self.mlp_text(self.norm2_text(x)))
        elif self.mlp_vl is None:
            x_text = x[:, : self.max_text_len]
            x_imag = x[:, self.max_text_len :]
            x_text = x_text + self.drop_path(self.gamma_2 * self.mlp_text(self.norm2_text(x_text)))
            x_imag = x_imag + self.drop_path(self.gamma_2 * self.mlp_imag(self.norm2_imag(x_imag)))
            x = torch.cat([x_text, x_imag], dim=1)
        else:
            x = x + self.drop_path(self.gamma_2 * self.mlp_vl(self.norm2_vl(x)))

        if modality_type == "eeg" and self.eeg_prompt is not None:
            x = x[:, : -self.prompt_len]
        elif modality_type == "eye" and self.eye_prompt is not None:
            x = x[:, : -self.prompt_len]
        return x


class MultiViewEmbedding(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, heads: int) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.heads = heads
        self.transform1 = nn.Linear(input_dim, output_dim)
        self.transform2 = nn.Linear(input_dim, output_dim * heads)
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm1d(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _ = x.shape
        x1 = self.transform1(x).unsqueeze(1).repeat(1, self.heads, 1)
        x2 = self.sigmoid(self.transform2(x)).reshape(b, self.heads, self.output_dim)
        x = torch.mul(x1, x2)
        x = self.bn(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x


class Fusion(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim, 1))
        self.softmax = nn.Softmax(1)

    def forward(self, eeg: torch.Tensor, eye: torch.Tensor) -> torch.Tensor:
        o1 = eeg @ self.weight
        o2 = eye @ self.weight
        o = torch.cat([o1, o2], dim=-1)
        alpha = self.softmax(o)
        return eeg * alpha[:, 0].unsqueeze(1) + eye * alpha[:, 1].unsqueeze(1)


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output.neg() * ctx.alpha, None


def default_maet_kwargs(
    *,
    num_classes: int = 7,
    domain_classes: int = 20,
    domain_generalization: bool = False,
) -> dict:
    return dict(
        eeg_dim=310,
        eye_dim=33,
        num_classes=num_classes,
        embed_dim=32,
        depth=3,
        eeg_seq_len=5,
        eye_seq_len=5,
        num_heads=4,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_abs_pos_emb=False,
        layer_scale_init_values=0.1,
        mixffn_start_layer_index=2,
        use_mean_pooling=False,
        domain_generalization=domain_generalization,
        num_domains=domain_classes,
        prompt=False,
        prompt_len=2,
    )


class MAET(nn.Module):
    """Multimodal Adaptive Emotion Transformer (EEG-only path used in this project)."""

    def __init__(
        self,
        eeg_dim: int = 310,
        eye_dim: int = 33,
        num_classes: int = 7,
        embed_dim: int = 32,
        depth: int = 3,
        eeg_seq_len: int = 5,
        eye_seq_len: int = 5,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer=nn.LayerNorm,
        use_abs_pos_emb: bool = False,
        layer_scale_init_values: float = 0.1,
        mixffn_start_layer_index: int = 2,
        use_mean_pooling: bool = False,
        domain_generalization: bool = False,
        num_domains: int = 19,
        prompt: bool = False,
        prompt_len: int = 2,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.eeg_seq_len = eeg_seq_len
        self.eye_seq_len = eye_seq_len
        self.mixffn_start_layer_index = mixffn_start_layer_index
        self.domain_generalization = domain_generalization
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.eeg_transform = MultiViewEmbedding(eeg_dim, embed_dim, eeg_seq_len)
        self.eye_transform = MultiViewEmbedding(eye_dim, embed_dim, eye_seq_len)
        self.eeg_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.eye_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.eeg_pos_embed = (
            nn.Parameter(torch.zeros(1, eeg_seq_len + 1, embed_dim)) if not use_abs_pos_emb else None
        )
        self.eye_pos_embed = (
            nn.Parameter(torch.zeros(1, eye_seq_len + 1, embed_dim)) if not use_abs_pos_emb else None
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.eeg_type_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.eye_type_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    with_mixffn=(i >= mixffn_start_layer_index),
                    layer_scale_init_values=layer_scale_init_values,
                    max_text_len=eeg_seq_len + 1,
                    prompt=prompt,
                    prompt_len=prompt_len,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.fusion = Fusion(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.head_eeg = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.head_eye = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.domain_classifier = None
        if domain_generalization:
            self.domain_classifier = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, num_domains),
            )

        if self.eeg_pos_embed is not None:
            _trunc_normal_(self.eeg_pos_embed, std=0.02)
        if self.eye_pos_embed is not None:
            _trunc_normal_(self.eye_pos_embed, std=0.02)
        _trunc_normal_(self.eeg_cls_token, std=0.02)
        _trunc_normal_(self.eye_cls_token, std=0.02)
        _trunc_normal_(self.eeg_type_embed, std=0.02)
        _trunc_normal_(self.eye_type_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            _trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def encode_eeg(self, eeg: torch.Tensor) -> torch.Tensor:
        """DE feature vector (B, 310) → pooled embedding (B, embed_dim)."""
        x = self.eeg_transform(eeg)
        cls = self.eeg_cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = x + self.eeg_type_embed.expand(x.size(0), x.size(1), -1)
        if self.eeg_pos_embed is not None:
            x = x + self.eeg_pos_embed.expand(x.size(0), -1, -1)
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x, modality_type="eeg")
        x = self.norm(x)
        return x[:, 1:].mean(dim=1)

    def forward(
        self,
        eeg: torch.Tensor | None = None,
        eye: torch.Tensor | None = None,
        alpha_: float = 0.0,
    ):
        if eeg is None and eye is None:
            raise ValueError("at least one of eeg/eye must be provided")

        if eeg is not None and eye is None:
            embed = self.encode_eeg(eeg)
            domain_out = None
            if self.domain_generalization and self.domain_classifier is not None:
                rev = ReverseLayerF.apply(embed, alpha_)
                domain_out = self.domain_classifier(rev)
            return self.head_eeg(embed), domain_out

        raise NotImplementedError("multimodal MAET forward not used in tone E1/E2 pipeline")
