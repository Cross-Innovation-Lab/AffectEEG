"""Seven paper-used task-adapted architectures, extracted from part2_class_style.

These are the project's clean-room task adaptations of architectural ideas;
they are not upstream implementations or faithful literature reproductions.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import torch
from torch import nn
from torch.nn import functional as F

@dataclass
class ClassStyleOutput:
    """Typed output shared by every class-style reconstruction model."""

    mel: torch.Tensor
    aux_losses: dict[str, torch.Tensor] = field(default_factory=dict)
    diagnostics: dict[str, torch.Tensor] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        """Return a legacy-friendly dictionary without detaching tensors."""
        return {
            "mel": self.mel,
            "aux_losses": self.aux_losses,
            "diagnostics": self.diagnostics,
        }

def _compatible_heads(dim: int, requested: int) -> int:
    for heads in range(min(dim, requested), 0, -1):
        if dim % heads == 0:
            return heads
    return 1

class MinimalStyleAdapter(nn.Module):
    """One low-rank, zero-initialized FiLM injection for external baselines."""

    def __init__(self, content_dim: int, style_dim: int = 128, rank: int = 4) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(content_dim, elementwise_affine=False)
        self.down = nn.Linear(style_dim, rank, bias=False)
        self.up = nn.Linear(rank, content_dim * 2, bias=True)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(
        self,
        content: torch.Tensor,
        style_token: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gamma, beta = self.up(F.gelu(self.down(style_token))).chunk(2, dim=-1)
        delta = self.norm(content) * gamma.unsqueeze(1) + beta.unsqueeze(1)
        return content + delta, delta

class _TemporalStem(nn.Module):
    def __init__(self, n_channels: int, model_dim: int, kernel_size: int = 7) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(n_channels)
        self.conv = nn.Conv1d(n_channels, model_dim, kernel_size, padding=kernel_size // 2)
        self.post = nn.Sequential(nn.GELU(), nn.GroupNorm(1, model_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B,T,C) -> (B,D,T) -> (B,T,D)
        return self.post(self.conv(self.norm(x).transpose(1, 2))).transpose(1, 2)

class _ResidualDilatedBlock(nn.Module):
    def __init__(self, dim: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.depthwise = nn.Conv1d(
            dim,
            dim,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=dim,
        )
        self.pointwise = nn.Conv1d(dim, dim * 2, kernel_size=1)
        self.out = nn.Sequential(nn.Conv1d(dim, dim, kernel_size=1), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.depthwise(self.norm(x).transpose(1, 2))
        value, gate = self.pointwise(hidden).chunk(2, dim=1)
        return x + self.out(torch.tanh(value) * torch.sigmoid(gate)).transpose(1, 2)

class _StateMixBlock(nn.Module):
    """A disclosed state-space-inspired gated convolution, not an S4/Mamba port."""

    def __init__(self, dim: int, dilation: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, dim * 2)
        self.state_conv = nn.Conv1d(
            dim,
            dim,
            kernel_size=5,
            padding=2 * dilation,
            dilation=dilation,
            groups=dim,
        )
        self.out_proj = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state, gate = self.in_proj(self.norm(x)).chunk(2, dim=-1)
        state = self.state_conv(state.transpose(1, 2)).transpose(1, 2)
        return x + self.out_proj(F.silu(state) * torch.sigmoid(gate))

class _QueryMelDecoder(nn.Module):
    def __init__(
        self,
        model_dim: int,
        n_mels: int,
        n_frames: int,
        num_heads: int = 4,
        depth: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        heads = _compatible_heads(model_dim, num_heads)
        self.frame_query = nn.Parameter(torch.empty(1, n_frames, model_dim))
        nn.init.normal_(self.frame_query, std=0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=depth)
        self.mel_projection = nn.Linear(model_dim, n_mels)

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        query = self.frame_query.expand(memory.size(0), -1, -1)
        frames = self.decoder(query, memory)  # (B,188,D)
        return self.mel_projection(frames).transpose(1, 2).contiguous()

class NeuroTalkContentCore(nn.Module):
    """Multi-receptive convolution plus BiGRU content core."""

    def __init__(self, n_channels: int, model_dim: int, dropout: float) -> None:
        super().__init__()
        branch_dim = model_dim // 4
        self.input_norm = nn.LayerNorm(n_channels)
        self.branches = nn.ModuleList([
            nn.Conv1d(n_channels, branch_dim, kernel, padding=kernel // 2)
            for kernel in (3, 5, 7, 11)
        ])
        merged_dim = branch_dim * 4
        self.to_model = nn.Linear(merged_dim, model_dim)
        self.temporal = nn.GRU(
            model_dim,
            model_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.out_dim = model_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        signal = self.input_norm(x).transpose(1, 2)
        branches = [F.gelu(branch(signal)) for branch in self.branches]
        sequence = self.to_model(torch.cat(branches, dim=1).transpose(1, 2))
        return self.temporal(sequence)[0]

class _DenseConvLayer(nn.Module):
    def __init__(self, in_dim: int, growth: int, dilation: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.conv = nn.Conv1d(
            in_dim,
            growth,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = F.gelu(self.conv(self.norm(x).transpose(1, 2))).transpose(1, 2)
        return torch.cat([x, update], dim=-1)

class ConvConcatContentCore(nn.Module):
    """Single dense concatenative temporal convolution core (no ensemble)."""

    def __init__(self, n_channels: int, model_dim: int, dropout: float) -> None:
        super().__init__()
        growth = max(model_dim // 4, 8)
        self.stem = _TemporalStem(n_channels, model_dim)
        dims = [model_dim + growth * index for index in range(4)]
        self.layers = nn.ModuleList([
            _DenseConvLayer(dim, growth, 2**index) for index, dim in enumerate(dims)
        ])
        self.projection = nn.Sequential(
            nn.LayerNorm(model_dim + growth * 4),
            nn.Linear(model_dim + growth * 4, model_dim),
            nn.Dropout(dropout),
        )
        self.out_dim = model_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.stem(x)
        for layer in self.layers:
            hidden = layer(hidden)
        return self.projection(hidden)

class CATWaveNetContentCore(nn.Module):
    """Gated dilated WaveNet content core for coarse-to-fine decoding."""

    def __init__(self, n_channels: int, model_dim: int, dropout: float) -> None:
        super().__init__()
        self.stem = _TemporalStem(n_channels, model_dim)
        self.blocks = nn.ModuleList([
            _ResidualDilatedBlock(model_dim, dilation, dropout)
            for dilation in (1, 2, 4, 8, 16, 1)
        ])
        self.out_dim = model_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.stem(x)
        for block in self.blocks:
            hidden = block(hidden)
        return hidden

class CATWaveNetMelDecoder(nn.Module):
    """Predict coarse mel from EEG memory, then refine it without GT mel."""

    def __init__(self, model_dim: int, n_mels: int, n_frames: int, dropout: float) -> None:
        super().__init__()
        self.coarse = _QueryMelDecoder(model_dim, 10, n_frames, depth=1, dropout=dropout)
        self.coarse_embedding = nn.Conv1d(10, model_dim, kernel_size=3, padding=1)
        self.cross_attention = nn.MultiheadAttention(
            model_dim,
            _compatible_heads(model_dim, 4),
            dropout=dropout,
            batch_first=True,
        )
        self.refine = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim * 2),
            nn.GELU(),
            nn.Linear(model_dim * 2, n_mels),
        )

    def forward(self, memory: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        coarse = self.coarse(memory)  # (B,10,188), predicted from EEG only
        query = self.coarse_embedding(coarse).transpose(1, 2)
        attended, _ = self.cross_attention(query, memory, memory, need_weights=False)
        mel = self.refine(query + attended).transpose(1, 2).contiguous()
        return mel, coarse

class FESDEContentCore(nn.Module):
    """State-space-inspired EEG encoder used by the FESDE task adaptation."""

    def __init__(self, n_channels: int, model_dim: int, dropout: float) -> None:
        super().__init__()
        self.stem = _TemporalStem(n_channels, model_dim, kernel_size=9)
        self.blocks = nn.ModuleList([
            _StateMixBlock(model_dim, dilation, dropout) for dilation in (1, 2, 4, 8)
        ])
        self.out_dim = model_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.stem(x)
        for block in self.blocks:
            hidden = block(hidden)
        return hidden

class FlowConnectorMelDecoder(nn.Module):
    """Invertible-coupling-inspired mel-state connector with no speech prior."""

    def __init__(self, model_dim: int, n_mels: int, n_frames: int, dropout: float) -> None:
        super().__init__()
        self.seed = _QueryMelDecoder(model_dim, model_dim, n_frames, depth=1, dropout=dropout)
        self.couplings = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(model_dim // 2, model_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv1d(model_dim, model_dim - model_dim // 2, kernel_size=1),
            )
            for _ in range(3)
        ])
        self.output = nn.Conv1d(model_dim, n_mels, kernel_size=1)

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        state = self.seed(memory)
        split = state.size(1) // 2
        for coupling in self.couplings:
            left, right = state[:, :split], state[:, split:]
            right = right + coupling(left)
            state = torch.cat([right, left], dim=1)
        return self.output(state)

class SSM2MelContentCore(nn.Module):
    """U-Net plus bidirectional state-mixing approximation without subject ESM."""

    def __init__(self, n_channels: int, model_dim: int, dropout: float) -> None:
        super().__init__()
        self.stem = _TemporalStem(n_channels, model_dim)
        self.down = nn.Conv1d(model_dim, model_dim, kernel_size=5, stride=2, padding=2)
        self.forward_state = nn.ModuleList([
            _StateMixBlock(model_dim, dilation, dropout) for dilation in (1, 2)
        ])
        self.backward_state = nn.ModuleList([
            _StateMixBlock(model_dim, dilation, dropout) for dilation in (1, 2)
        ])
        self.skip_mix = nn.Linear(model_dim * 2, model_dim)
        self.out_dim = model_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.stem(x)  # (B,T,D)
        downsampled = F.gelu(self.down(skip.transpose(1, 2))).transpose(1, 2)
        forward = downsampled
        backward = torch.flip(downsampled, dims=[1])
        for forward_block, backward_block in zip(self.forward_state, self.backward_state):
            forward = forward_block(forward)
            backward = backward_block(backward)
        hidden = 0.5 * (forward + torch.flip(backward, dims=[1]))
        up = F.interpolate(
            hidden.transpose(1, 2), size=skip.size(1), mode="linear", align_corners=False
        ).transpose(1, 2)
        return self.skip_mix(torch.cat([skip, up], dim=-1))

class DynamicMultiscaleContentCore(nn.Module):
    """Dynamic multi-scale fusion approximation for DMF2Mel."""

    def __init__(self, n_channels: int, model_dim: int, dropout: float) -> None:
        super().__init__()
        self.stem = _TemporalStem(n_channels, model_dim)
        self.branches = nn.ModuleList([
            nn.Conv1d(
                model_dim,
                model_dim,
                kernel_size=kernel,
                padding=(kernel // 2) * dilation,
                dilation=dilation,
                groups=model_dim,
            )
            for kernel, dilation in ((3, 1), (5, 1), (3, 2), (3, 4))
        ])
        self.router = nn.Linear(model_dim, len(self.branches))
        self.fusion = nn.Linear(model_dim, model_dim)
        self.state = nn.ModuleList([
            _StateMixBlock(model_dim, dilation, dropout) for dilation in (1, 2, 4)
        ])
        self.out_dim = model_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.stem(x)
        branch_values = torch.stack([
            F.gelu(branch(hidden.transpose(1, 2))).transpose(1, 2)
            for branch in self.branches
        ], dim=1)  # (B,scale,T,D)
        weights = torch.softmax(self.router(hidden.mean(dim=1)), dim=-1)
        mixed = (branch_values * weights[:, :, None, None]).sum(dim=1)
        hidden = hidden + self.fusion(mixed)
        for block in self.state:
            hidden = block(hidden)
        return hidden

def _linear_spline_basis(n_mels: int, n_knots: int) -> torch.Tensor:
    mel_positions = torch.linspace(0.0, n_knots - 1.0, n_mels).unsqueeze(1)
    knot_positions = torch.arange(n_knots, dtype=torch.float32).unsqueeze(0)
    basis = (1.0 - (mel_positions - knot_positions).abs()).clamp_min(0.0)
    return basis / basis.sum(dim=1, keepdim=True).clamp_min(1e-8)

class AdaptiveSplineMelDecoder(nn.Module):
    """SplineMap/AGKAN-inspired clean-room frequency mapper for DMF2Mel."""

    def __init__(
        self,
        model_dim: int,
        n_mels: int,
        n_frames: int,
        dropout: float,
        n_knots: int = 25,
    ) -> None:
        super().__init__()
        self.knot_decoder = _QueryMelDecoder(
            model_dim, n_knots, n_frames, depth=2, dropout=dropout
        )
        self.register_buffer(
            "linear_basis", _linear_spline_basis(n_mels, n_knots), persistent=True
        )
        self.basis_residual = nn.Parameter(torch.zeros(n_mels, n_knots))
        self.temporal_refine = nn.Conv1d(n_mels, n_mels, kernel_size=3, padding=1, groups=n_mels)

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        knots = self.knot_decoder(memory)  # (B,25,188)
        weights = self.linear_basis + 0.01 * torch.tanh(self.basis_residual)
        mel = torch.einsum("mk,bkf->bmf", weights, knots)
        return mel + self.temporal_refine(mel)

class NeuroSonicContentCore(nn.Module):
    """Temporal content encoder for deterministic conditional-flow decoding."""

    def __init__(self, n_channels: int, model_dim: int, dropout: float) -> None:
        super().__init__()
        self.stem = _TemporalStem(n_channels, model_dim, kernel_size=11)
        heads = _compatible_heads(model_dim, 4)
        layer = nn.TransformerEncoderLayer(
            model_dim,
            heads,
            model_dim * 4,
            dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=3)
        self.out_dim = model_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.stem(x))

class DeterministicFlowMelDecoder(nn.Module):
    """Fixed-step Heun conditional flow with a stable evaluation noise buffer."""

    def __init__(
        self,
        model_dim: int,
        n_mels: int,
        n_frames: int,
        dropout: float,
        solver_steps: int = 4,
        noise_seed: int = 20260729,
    ) -> None:
        super().__init__()
        self.solver_steps = int(solver_steps)
        if self.solver_steps <= 0:
            raise ValueError("solver_steps must be positive")
        self.solver_name = "heun"
        self.ode_function_evaluations = 2 * self.solver_steps
        generator = torch.Generator().manual_seed(noise_seed)
        noise = torch.randn(1, n_frames, model_dim, generator=generator) * 0.02
        self.register_buffer("evaluation_noise", noise, persistent=True)
        self.condition = nn.MultiheadAttention(
            model_dim,
            _compatible_heads(model_dim, 4),
            dropout=dropout,
            batch_first=True,
        )
        self.time_embedding = nn.Sequential(nn.Linear(1, model_dim), nn.SiLU())
        self.velocity = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim * 2),
            nn.SiLU(),
            nn.Linear(model_dim * 2, model_dim),
        )
        self.output = nn.Linear(model_dim, n_mels)

    def _drift(
        self, state: torch.Tensor, memory: torch.Tensor, time_value: float
    ) -> torch.Tensor:
        conditioned, _ = self.condition(state, memory, memory, need_weights=False)
        time = state.new_full((state.size(0), 1), float(time_value))
        return self.velocity(
            state + conditioned + self.time_embedding(time).unsqueeze(1)
        )

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        state = self.evaluation_noise.expand(memory.size(0), -1, -1).to(memory)
        step_size = 1.0 / self.solver_steps
        for step in range(self.solver_steps):
            time_value = step * step_size
            first = self._drift(state, memory, time_value)
            predictor = state + step_size * first
            second = self._drift(predictor, memory, time_value + step_size)
            state = state + 0.5 * step_size * (first + second)
        return self.output(state).transpose(1, 2).contiguous()

def _build_external_core(
    slug: str,
    n_channels: int,
    model_dim: int,
    dropout: float,
) -> nn.Module:
    core_types: dict[str, type[nn.Module]] = {
        "neurotalk": NeuroTalkContentCore,
        "convconcatnet": ConvConcatContentCore,
        "cat_wavenet": CATWaveNetContentCore,
        "fesde": FESDEContentCore,
        "ssm2mel": SSM2MelContentCore,
        "dmf2mel": DynamicMultiscaleContentCore,
        "neurosonic": NeuroSonicContentCore,
    }
    return core_types[slug](n_channels, model_dim, dropout)

def _build_external_decoder(
    slug: str,
    model_dim: int,
    n_mels: int,
    n_frames: int,
    dropout: float,
) -> nn.Module:
    if slug == "cat_wavenet":
        return CATWaveNetMelDecoder(model_dim, n_mels, n_frames, dropout)
    if slug == "fesde":
        return FlowConnectorMelDecoder(model_dim, n_mels, n_frames, dropout)
    if slug == "neurosonic":
        return DeterministicFlowMelDecoder(model_dim, n_mels, n_frames, dropout)
    if slug == "dmf2mel":
        return AdaptiveSplineMelDecoder(model_dim, n_mels, n_frames, dropout)
    return _QueryMelDecoder(model_dim, n_mels, n_frames, depth=2, dropout=dropout)
