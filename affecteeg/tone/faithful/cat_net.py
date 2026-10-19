import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x):
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -grad_output

class GradientReversal(nn.Module):
    def forward(self, x):
        return GradientReversalFunction.apply(x)

class ChannelAttention(nn.Module):
    """SE-style channel attention: Global Avg + Max pool → shared MLP → scale."""
    def __init__(self, channel_dim, reduction=8):
        super().__init__()
        self.fc1 = nn.Linear(channel_dim, channel_dim // reduction)
        self.fc2 = nn.Linear(channel_dim // reduction, channel_dim)

    def forward(self, x):
        # x: (B, C, T) — channels first
        avg_pool = x.mean(dim=2)
        max_pool = x.max(dim=2).values

        avg_out = torch.sigmoid(self.fc2(F.relu(self.fc1(avg_pool))))
        max_out = torch.sigmoid(self.fc2(F.relu(self.fc1(max_pool))))

        scale = (avg_out + max_out).unsqueeze(2)  # (B, C, 1)
        self.last_scale = scale.squeeze(2).detach()  # (B, C) for analysis
        return x * scale

class B3ModelNoAdapter(nn.Module):
    """
    双分支（EEG + EMG）模型：
      Conv1D → ChannelAttention → BiLSTM → Cross-Attention → Pooling → 融合分类 + 域对抗。

    输入维度约定（与原 TF 版一致）：
      eeg_input: (B, timesteps_eeg, eeg_channels * 2)
      emg_input: (B, timesteps_emg, emg_channels * 2)
    """
    def __init__(self, timesteps_eeg, eeg_channels, timesteps_emg, emg_channels,
                 num_classes, domain_classes=10, dropout_rate=0.5,
                 eeg_input_indices=None):
        super().__init__()

        indices = torch.as_tensor(
            [] if eeg_input_indices is None else eeg_input_indices,
            dtype=torch.long,
        )
        self.register_buffer("eeg_input_indices", indices, persistent=True)

        # ======== EEG 分支 ========
        self.eeg_conv1 = nn.Conv1d(eeg_channels * 2, 64, kernel_size=10, padding='same')
        self.eeg_bn1 = nn.BatchNorm1d(64)
        self.eeg_conv2 = nn.Conv1d(64, 128, kernel_size=5, padding='same')
        self.eeg_bn2 = nn.BatchNorm1d(128)
        self.eeg_pool = nn.MaxPool1d(kernel_size=2)
        self.eeg_channel_attn = ChannelAttention(128)
        self.eeg_lstm1 = nn.LSTM(128, 64, batch_first=True, bidirectional=True)
        self.eeg_lstm2 = nn.LSTM(128, 64, batch_first=True, bidirectional=True)
        self.eeg_ln = nn.LayerNorm(128)
        self.eeg_dropout = nn.Dropout(dropout_rate)

        # ======== EMG 分支 ========
        self.emg_conv1 = nn.Conv1d(emg_channels * 2, 64, kernel_size=10, padding='same')
        self.emg_bn1 = nn.BatchNorm1d(64)
        self.emg_conv2 = nn.Conv1d(64, 128, kernel_size=5, padding='same')
        self.emg_bn2 = nn.BatchNorm1d(128)
        self.emg_pool = nn.MaxPool1d(kernel_size=2)
        self.emg_channel_attn = ChannelAttention(128)
        self.emg_lstm1 = nn.LSTM(128, 64, batch_first=True, bidirectional=True)
        self.emg_lstm2 = nn.LSTM(128, 64, batch_first=True, bidirectional=True)
        self.emg_ln = nn.LayerNorm(128)
        self.emg_dropout = nn.Dropout(dropout_rate)

        # ======== Cross-Attention ========
        self.eeg_pre_norm = nn.LayerNorm(128)
        self.emg_pre_norm = nn.LayerNorm(128)
        self.cross_attn_eeg = nn.MultiheadAttention(128, num_heads=4, batch_first=True)
        self.cross_attn_emg = nn.MultiheadAttention(128, num_heads=4, batch_first=True)
        self.eeg_post_norm1 = nn.LayerNorm(128)
        self.emg_post_norm1 = nn.LayerNorm(128)
        self.eeg_post_norm2 = nn.LayerNorm(128)
        self.emg_post_norm2 = nn.LayerNorm(128)

        # ======== 融合 + 分类 ========
        # pooled: avg(128) + max(128) per modality → 256 * 2 = 512
        self.fused_fc = nn.Linear(512, 128)
        self.fused_bn = nn.BatchNorm1d(128)
        self.fused_dropout = nn.Dropout(dropout_rate)
        self.class_head = nn.Linear(128, num_classes)

        # ======== 域对抗分支 (GRL) ========
        self.grl = GradientReversal()
        self.domain_fc = nn.Linear(128, 64)
        self.domain_dropout = nn.Dropout(0.3)
        self.domain_head = nn.Linear(64, domain_classes)

    def forward(self, eeg_input, emg_input, return_features=False,
                return_attention=False):
        """
        Args:
            eeg_input: (B, T_eeg, C_eeg)  channels-last (与 numpy 数据一致)
            emg_input: (B, T_emg, C_emg)
            return_features: 若为 True，返回融合特征而非分类输出
            return_attention: 若为 True，额外返回注意力权重字典
        """
        if self.eeg_input_indices.numel():
            eeg_input = eeg_input.index_select(-1, self.eeg_input_indices)

        # --- EEG 分支 ---
        eeg = eeg_input.permute(0, 2, 1)                       # (B, C, T)
        eeg = F.relu(self.eeg_bn1(self.eeg_conv1(eeg)))
        eeg = F.relu(self.eeg_bn2(self.eeg_conv2(eeg)))
        eeg = self.eeg_pool(eeg)
        eeg = self.eeg_channel_attn(eeg)

        eeg = eeg.permute(0, 2, 1)                             # (B, T', 128)
        eeg, _ = self.eeg_lstm1(eeg)
        eeg, _ = self.eeg_lstm2(eeg)
        eeg = self.eeg_ln(eeg)
        eeg = self.eeg_dropout(eeg)

        # --- EMG 分支 ---
        emg = emg_input.permute(0, 2, 1)
        emg = F.relu(self.emg_bn1(self.emg_conv1(emg)))
        emg = F.relu(self.emg_bn2(self.emg_conv2(emg)))
        emg = self.emg_pool(emg)
        emg = self.emg_channel_attn(emg)

        emg = emg.permute(0, 2, 1)
        emg, _ = self.emg_lstm1(emg)
        emg, _ = self.emg_lstm2(emg)
        emg = self.emg_ln(emg)
        emg = self.emg_dropout(emg)

        # --- Cross-Attention ---
        eeg_norm = self.eeg_pre_norm(eeg)
        emg_norm = self.emg_pre_norm(emg)
        attn_eeg, cross_w_eeg = self.cross_attn_eeg(
            eeg_norm, emg_norm, emg_norm, need_weights=return_attention)
        attn_emg, cross_w_emg = self.cross_attn_emg(
            emg_norm, eeg_norm, eeg_norm, need_weights=return_attention)
        attn_eeg = self.eeg_post_norm2(self.eeg_post_norm1(attn_eeg))
        attn_emg = self.emg_post_norm2(self.emg_post_norm1(attn_emg))

        # --- Global Average + Max Pooling ---
        pooled_eeg = torch.cat([attn_eeg.mean(dim=1), attn_eeg.max(dim=1).values], dim=1)
        pooled_emg = torch.cat([attn_emg.mean(dim=1), attn_emg.max(dim=1).values], dim=1)

        # --- 融合 ---
        fused = torch.cat([pooled_eeg, pooled_emg], dim=1)     # (B, 512)
        fused = F.relu(self.fused_fc(fused))
        fused = self.fused_bn(fused)
        fused = self.fused_dropout(fused)

        attn_dict = None
        if return_attention:
            attn_dict = {
                "eeg_channel_attn": self.eeg_channel_attn.last_scale,  # (B, 128)
                "emg_channel_attn": self.emg_channel_attn.last_scale,
                "cross_attn_eeg": cross_w_eeg,   # (B, T_q, T_k) or None
                "cross_attn_emg": cross_w_emg,
            }

        if return_features:
            return (fused, attn_dict) if return_attention else fused

        # --- 分类头 ---
        class_output = self.class_head(fused)

        # --- 域对抗头 ---
        reversed_feat = self.grl(fused)
        domain_output = F.relu(self.domain_fc(reversed_feat))
        domain_output = self.domain_dropout(domain_output)
        domain_output = self.domain_head(domain_output)

        if return_attention:
            return class_output, domain_output, attn_dict
        return class_output, domain_output
