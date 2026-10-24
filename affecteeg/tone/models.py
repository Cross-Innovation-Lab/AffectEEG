"""Ten main-experiment backbones, all initialized from scratch.

Vendored architecture code and electrode coordinates support local initialization.
The adapters preserve the upstream model interfaces.
"""
from pathlib import Path
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from .faithful.base import ToneBaselineWrapper
from .protocol import BACKBONES, EEG20


class PatchBackbone(ToneBaselineWrapper):
    def __init__(self):
        super().__init__(20, 0, 4)
        self.timesteps = 500

    def signal(self, eeg, length):
        if eeg.shape[1:] != (500, 20):
            raise ValueError(f"Expected (B,500,20), got {tuple(eeg.shape)}")
        return F.interpolate(eeg.transpose(1, 2), size=length,
                             mode='linear', align_corners=False)


class CBraMod(PatchBackbone):
    def __init__(self):
        super().__init__()
        from vendor.tone.cbramod.cbramod import CBraMod as Encoder
        self.backbone = Encoder(in_dim=200, out_dim=200, d_model=200,
                                dim_feedforward=800, seq_len=4, n_layer=12, nhead=8)
        self.backbone.proj_out = nn.Identity()
        self.classifier = nn.Linear(200, 4)

    def adapt_eeg(self, eeg):
        patches = self.signal(eeg, 800).reshape(-1, 20, 4, 200)
        return patches, None

    def classify(self, eeg, emg):
        patches, mask = self.adapt_eeg(eeg)
        return self.classifier(self.backbone(patches, mask=mask).mean((1, 2)))


class STEEGFormer(PatchBackbone):
    def __init__(self):
        super().__init__()
        from vendor.tone.st_eegformer import vit_small_patch16
        self.backbone = vit_small_patch16(num_classes=4, global_pool=True,
            drop_rate=.1, drop_path_rate=.1, attn_drop_rate=.1, proj_drop_rate=.1)
        # Scratch training learns these channel embeddings jointly from epoch 1.
        self.register_buffer('channel_indices', torch.arange(20))
        # Freeze parameters outside the forward_features computation.
        self.backbone.pos_embed.requires_grad_(False)
        if hasattr(self.backbone, 'fc_norm'):
            self.backbone.fc_norm.requires_grad_(False)

    def adapt_eeg(self, eeg):
        return self.signal(eeg, 512)

    def classify(self, eeg, emg):
        x = self.adapt_eeg(eeg)
        ids = self.channel_indices.expand(x.shape[0], -1)
        return self.backbone.head(self.backbone.head_drop(self.backbone.forward_features(x, ids)))


class EEGDeformer(PatchBackbone):
    def __init__(self):
        super().__init__()
        from vendor.tone.eeg_deformer import Deformer
        self.backbone = Deformer(num_chan=20, num_time=500, temporal_kernel=13,
            num_kernel=64, num_classes=4, depth=4, heads=16, mlp_dim=16,
            dim_head=16, dropout=.4)

    def classify(self, eeg, emg):
        return self.backbone(eeg.transpose(1, 2))


class EEGMamba(PatchBackbone):
    def __init__(self):
        super().__init__()
        try:
            from vendor.tone.eegmamba.eegmamba import EEGMamba as Encoder
        except ImportError as exc:
            raise ImportError('EEGMamba requires the CUDA mamba-ssm and causal-conv1d dependencies; see README.') from exc
        self.backbone = Encoder(in_dim=200, out_dim=200, d_model=200,
                                dim_feedforward=800, seq_len=4, n_layer=12, nhead=8)
        self.backbone.proj_out = nn.Identity()
        self.classifier = nn.Linear(200, 4)

    def adapt_eeg(self, eeg):
        return self.signal(eeg, 800).reshape(-1, 20, 4, 200)

    def classify(self, eeg, emg):
        if eeg.device.type != 'cuda':
            raise RuntimeError('The official EEGMamba Mamba2 kernels require a CUDA device.')
        return self.classifier(self.backbone(self.adapt_eeg(eeg)).mean((1, 2)))


class REVEHead(nn.Module):
    def __init__(self, dimension=512):
        super().__init__()
        from vendor.tone.reve.modeling_reve import RMSNorm
        self.embed_dim = dimension
        self.cls_query_token = nn.Parameter(torch.randn(1, 1, dimension))
        self.linear_head = nn.Sequential(RMSNorm(dimension), nn.Dropout(.4), nn.Linear(dimension, 4))

    def forward(self, value):
        value = value.flatten(1, 2)
        query = self.cls_query_token.expand(value.shape[0], -1, -1)
        attention = torch.softmax(query @ value.transpose(-1, -2) / math.sqrt(self.embed_dim), -1)
        return self.linear_head((attention @ value).squeeze(1))


class REVE(PatchBackbone):
    def __init__(self):
        super().__init__()
        from vendor.tone.reve.configuration_reve import ReveConfig
        from vendor.tone.reve.modeling_reve import Reve
        self.backbone = Reve(ReveConfig(embed_dim=512, depth=22, heads=8, head_dim=64,
            mlp_dim_ratio=2.66, use_geglu=True, freqs=4, noise_ratio=.0025,
            patch_size=200, patch_overlap=20))
        self.head = REVEHead()
        with torch.no_grad():
            self.head.cls_query_token.copy_(self.backbone.cls_query_token)
        path = Path(__file__).resolve().parents[2] / 'vendor/tone/reve/positions.json'
        values = json.loads(path.read_text())
        values = values.get('positions', values)
        self.register_buffer('electrode_positions', torch.tensor([values[c] for c in EEG20], dtype=torch.float32))

    def adapt_eeg(self, eeg):
        return self.signal(eeg, 800)

    def classify(self, eeg, emg):
        return self.head(self.backbone(self.adapt_eeg(eeg), self.electrode_positions.expand(eeg.shape[0], -1, -1)))


def build_model(backbone):
    if backbone == 'cat_net':
        from .faithful.cat_net import B3ModelNoAdapter
        return B3ModelNoAdapter(499, 5, 499, 5, 4, domain_classes=10, dropout_rate=.4)
    if backbone == 'eegnet':
        from .faithful.eegnet import FaithfulEEGNet
        return FaithfulEEGNet(500, 20, 0, 4)
    if backbone == 'atcnet':
        from .faithful.atcnet import FaithfulATCNet
        return FaithfulATCNet(500, 20, 0, 4)
    if backbone == 'eeg_conformer':
        from .faithful.eeg_conformer import FaithfulEEGConformer
        return FaithfulEEGConformer(500, 20, 0, 4)
    if backbone == 'rgnn':
        from .faithful.rgnn import FaithfulRGNN
        return FaithfulRGNN(500, 20, 0, 4, channel_names=EEG20)
    classes = dict(cbramod=CBraMod, st_eegformer=STEEGFormer,
                   eeg_deformer=EEGDeformer, eegmamba=EEGMamba, reve=REVE)
    if backbone not in classes:
        raise ValueError(f'Unknown backbone {backbone}; choose from {BACKBONES}')
    return classes[backbone]()


def head_modules(backbone, model):
    if backbone in ('cbramod', 'eegmamba'):
        return (model.classifier,)
    if backbone == 'st_eegformer':
        return (model.backbone.head,)
    if backbone == 'reve':
        return (model.head,)
    return ()
