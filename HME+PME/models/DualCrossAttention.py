import torch
import torch.nn as nn
import torch.nn.functional as F


# 2-D Cross attention
class CrossAttention(nn.Module):
    """
    Generalized Scaled Dot-Product Cross Attention
    - enc1_feats: Query source, shape (B, C1, H1, W1)
    - enc2_feats: Key/Value source, shape (B, C2, H2, W2)
    Outputs:
        - context: (B, attn_dim)
        - attn_map: (B, H1, W1) ← query 위치 기준의 attention map
    """
    def __init__(self, enc1_dim, enc2_dim, attn_dim):
        super().__init__()
        self.query_proj = nn.Conv2d(enc1_dim, attn_dim, kernel_size=1)
        self.key_proj   = nn.Conv2d(enc2_dim, attn_dim, kernel_size=1)
        self.value_proj = nn.Conv2d(enc2_dim, attn_dim, kernel_size=1)
        self.resample_conv = nn.Conv2d(enc2_dim, enc2_dim, kernel_size=1)  # lightweight resampler
        self.scale = attn_dim ** 0.5

    def forward(self, enc1_feats, enc2_feats):
        B, _, H1, W1 = enc1_feats.size()
        B2, _, H2, W2 = enc2_feats.size()

        # ⚠️ Resample enc2_feats to match spatial size of enc1_feats
        if (H1, W1) != (H2, W2):
            enc2_feats = self.resample_conv(enc2_feats)

            try:
                # ✅ adaptive avg pool 우선 시도 (모든 디바이스 공통)
                enc2_feats = F.adaptive_avg_pool2d(enc2_feats, output_size=(H1, W1))
            except RuntimeError:
                # ✅ 실패 시 fallback → interpolate
                enc2_feats = F.interpolate(enc2_feats, size=(H1, W1), mode='bilinear', align_corners=False)
        else:
            # ✅ 동일한 경우에도 pooling으로 보정
            enc2_feats = F.adaptive_avg_pool2d(enc2_feats, output_size=(H1, W1))

        # ✅ Query from enc1_feats
        query = self.query_proj(enc1_feats)                      # (B, D, H1, W1)
        query = query.flatten(2).transpose(1, 2)                 # (B, H1*W1, D)

        # ✅ Key/Value from enc2_feats
        key   = self.key_proj(enc2_feats).flatten(2)             # (B, D, H1*W1)
        value = self.value_proj(enc2_feats).flatten(2).transpose(1, 2)  # (B, H1*W1, D)

        # ✅ Scaled Dot-Product Attention
        attn_scores  = torch.bmm(query, key) / self.scale        # (B, H1*W1, H1*W1)
        attn_weights = F.softmax(attn_scores, dim=-1)            # (B, H1*W1, H1*W1)

        # ✅ Context vector
        context = torch.bmm(attn_weights, value)                 # (B, H1*W1, D)
        context = context.mean(dim=1)                            # (B, D)

        # ✅ Attention map (mean over attention targets)
        attn_map = attn_weights.mean(dim=2).view(B, H1, W1)      # (B, H1, W1)

        return context, attn_map



class DualCrossAttention(nn.Module):
    """
    Applies bidirectional cross-attention between PME and HME encoders.
    - feat_pme: (B, Cp, H1, W1)
    - feat_hme: (B, Ch, H2, W2)
    Outputs:
        - context_p_to_h: (B, attn_dim)
        - context_h_to_p: (B, attn_dim)
        - attn_map_p_to_h: (B, H, W)
        - attn_map_h_to_p: (B, H, W)
    """

    def __init__(self, pme_dim, hme_dim, attn_dim):
        super().__init__()
        self.pme_to_hme = CrossAttention(pme_dim, hme_dim, attn_dim)
        self.hme_to_pme = CrossAttention(hme_dim, pme_dim, attn_dim)

    def forward(self, feat_pme, feat_hme):
        context_p_to_h, attn_p_to_h = self.pme_to_hme(feat_pme, feat_hme)
        context_h_to_p, attn_h_to_p = self.hme_to_pme(feat_hme, feat_pme)
        return context_p_to_h, context_h_to_p, attn_p_to_h, attn_h_to_p
