import torch
import torch.nn as nn
import torch.nn.functional as F

class CoverageAttention(nn.Module):
    """
    Coverage-based Additive Attention module as described in the paper.
    Inputs:
    - decoder_hidden: (B, H)         # h_{t-1}
    - encoder_feats: (B, C, Hf, Wf)  # F_{u,v}
    - prev_coverage: (B, 1, Hf, Wf)  # sum of past attention maps (optional)
    
    Outputs:
    - context: (B, C)
    - attention_weights: (B, Hf, Wf)
    - updated_coverage: (B, 1, Hf, Wf)
    """
    def __init__(self, hidden_dim, encoder_dim, attn_dim):
        super().__init__()
        self.W_h = nn.Linear(hidden_dim, attn_dim)
        self.W_F = nn.Conv2d(encoder_dim, attn_dim, kernel_size=1)
        self.W_cov = nn.Conv2d(1, attn_dim, kernel_size=3, padding=1)
        self.v_att = nn.Linear(attn_dim, 1)

    def forward(self, decoder_hidden, encoder_feats, prev_coverage=None):
        B, C, Hf, Wf = encoder_feats.size()

        if prev_coverage is None:
            prev_coverage = torch.zeros(B, 1, Hf, Wf, device=encoder_feats.device)

        # Linear transforms
        decoder_proj = self.W_h(decoder_hidden).unsqueeze(2).unsqueeze(3)   # (B, attn_dim, 1, 1)
        encoder_proj = self.W_F(encoder_feats)                              # (B, attn_dim, Hf, Wf)
        coverage_proj = self.W_cov(prev_coverage)                           # (B, attn_dim, Hf, Wf)

        # Energy computation
        e = self.v_att(
            torch.tanh(decoder_proj + encoder_proj + coverage_proj)
        ).squeeze(1)  # (B, Hf, Wf)

        # Attention weights
        alpha = F.softmax(e.view(B, -1), dim=-1).view(B, Hf, Wf)  # (B, Hf, Wf)

        # Context vector (weighted sum over spatial features)
        alpha_exp = alpha.unsqueeze(1)                            # (B, 1, Hf, Wf)
        context = torch.sum(encoder_feats * alpha_exp, dim=(2, 3))  # (B, C)

        # Updated coverage
        updated_coverage = prev_coverage + alpha_exp              # (B, 1, Hf, Wf)

        return context, alpha, updated_coverage
