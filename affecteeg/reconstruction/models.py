"""The manuscript Base and complete AffectEEG reconstruction models."""
from __future__ import annotations
import torch
from torch import nn
from .backbones import ClassStyleOutput, MinimalStyleAdapter, _build_external_core, _build_external_decoder
from .arra import StrongBoundedAdapter, evaluation_without_grad, grouped_covariance, DeterministicSSM2MelCore

BACKBONES = ('neurotalk', 'convconcatnet', 'fesde', 'ssm2mel', 'dmf2mel', 'neurosonic', 'cat_wavenet')


class ConsistentEncoderTransformerLayer(nn.TransformerEncoderLayer):
    def forward(self, *args, **kwargs):
        enabled = torch.backends.mha.get_fastpath_enabled()
        try:
            torch.backends.mha.set_fastpath_enabled(False)
            return super().forward(*args, **kwargs)
        finally:
            torch.backends.mha.set_fastpath_enabled(enabled)


class ReplayableGRU(nn.GRU):
    def forward(self, input, hx=None):
        if self.training and self.dropout > 0 and self.num_layers > 1:
            with torch.backends.cudnn.flags(enabled=False):
                return super().forward(input, hx)
        return super().forward(input, hx)


class ReconstructionModel(nn.Module):
    def __init__(self, backbone, variant='base', feature_statistics=None, pool_statistics=None):
        super().__init__()
        if backbone not in BACKBONES or variant not in ('base', 'affecteeg'):
            raise ValueError((backbone, variant))
        self.backbone, self.variant = backbone, variant
        self.content_encoder = _build_external_core(backbone, 59, 256, .1)
        self.decoder = _build_external_decoder(backbone, 256, 100, 188, .1)
        if backbone == 'ssm2mel':
            self.content_encoder = DeterministicSSM2MelCore(self.content_encoder)
        for module in self.content_encoder.modules():
            if type(module) is nn.TransformerEncoderLayer:
                module.__class__ = ConsistentEncoderTransformerLayer
            if type(module) is nn.GRU and module.dropout > 0 and module.num_layers > 1:
                module.__class__ = ReplayableGRU
        self.fusion = None
        self.adapter = None
        if variant == 'affecteeg':
            # ARRA statistics depend on this model's random initialization. A new
            # model is constructed first, then initialize_adapter fits them.
            self.fusion = MinimalStyleAdapter(256, 128, rank=4)
            if feature_statistics is not None:
                self.initialize_adapter(feature_statistics, pool_statistics)

    def initialize_adapter(self, feature_statistics, pool_statistics):
        self.adapter = StrongBoundedAdapter(feature_statistics, pool_statistics).to(next(self.parameters()).device)

    def encode_auxiliary(self, eeg):
        with evaluation_without_grad(self.content_encoder):
            return self.content_encoder(eeg).contiguous()

    def forward_memory(self, memory, prototype=None):
        diagnostics = {}
        if self.variant == 'affecteeg':
            if self.adapter is None or prototype is None:
                raise ValueError('AffectEEG requires fitted training statistics and an affect prototype mixture')
            clean = self.adapter(memory)
            detached = memory.detach()
            diagnostics['preservation'] = self.adapter.preservation(detached, self.adapter(detached))
            memory, residual = self.fusion(clean, prototype)
            diagnostics['fusion_residual'] = residual
        decoded = self.decoder(memory)
        if isinstance(decoded, tuple):
            decoded, diagnostics['coarse_mel'] = decoded
        return ClassStyleOutput(mel=decoded, diagnostics=diagnostics)

    def forward(self, eeg, prototype=None):
        return self.forward_memory(self.content_encoder(eeg), prototype)

    def auxiliary_covariance(self, eeg, teacher_targets, groups):
        memory = self.encode_auxiliary(eeg)
        return grouped_covariance(self.adapter.view(self.adapter(memory)), teacher_targets.detach(), groups, group_size=8)
