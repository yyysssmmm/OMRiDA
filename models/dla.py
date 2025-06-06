import torch
import torch.nn as nn
from .encoder import DenseNetEncoder
from .decoder import Decoder
from .DualCrossAttention import DualCrossAttention  # 수정된 이름 주의!

class DLAModel(nn.Module):
    """
    Dual-Cross-Attention 기반 MER 모델.
    - PME 이미지와 HME 이미지를 각각 인코딩
    - 양방향 Cross Attention 수행
    - PME 디코더는 HME context 사용, HME 디코더는 PME context 사용
    """

    def __init__(self, vocab_size, emb_dim=64, enc_dim=512, attn_dim=256, hidden_dim=256):
        super().__init__()
        self.encoder_pme = DenseNetEncoder(out_channels=enc_dim)
        self.encoder_hme = DenseNetEncoder(out_channels=enc_dim)

        self.cross_attention = DualCrossAttention(
            pme_dim=enc_dim, hme_dim=enc_dim, attn_dim=attn_dim
        )

        self.decoder_pme = Decoder(vocab_size, emb_dim, attn_dim, hidden_dim)
        self.decoder_hme = Decoder(vocab_size, emb_dim, attn_dim, hidden_dim)

        self.hidden_dim = hidden_dim

    def forward(self, images_pme, images_hme, tgt_seq_pme, tgt_seq_hme, teacher_forcing_ratio=0.5, pme_only=False):
        """
        Args:
            images_pme: (B, 3, H, W)
            images_hme: (B, 3, H, W)
            tgt_seq_pme: (B, T)
            tgt_seq_hme: (B, T)
            pme_only: True이면 HME 인코더, 디코더, CrossAttention을 생략 (unpaired PME 학습에 사용)

        Returns:
            logits_pme: (B, T-1, vocab_size)
            logits_hme: (B, T-1, vocab_size) 또는 None
            context_p_to_h: (B, D)
            context_h_to_p: (B, D) 또는 None
        """
        B, T = tgt_seq_pme.shape
        device = images_pme.device
        hidden_pme = torch.zeros(1, B, self.hidden_dim, device=device)
        hidden_hme = torch.zeros(1, B, self.hidden_dim, device=device)

        # 🔁 인코딩 (PME → Encoder)
        feat_pme = self.encoder_pme(images_pme)  # (B, C, H', W')

        if not pme_only:
            feat_hme = self.encoder_hme(images_hme)  # (B, C, H', W')

            # 🔁 양방향 Cross Attention (feat_pme ↔ feat_hme)
            context_p_to_h, context_h_to_p, _, _ = self.cross_attention(feat_pme, feat_hme)
        else:
            # 🔁 모델 순전파 (unpaired PME → only decoder loss)
            context_p_to_h = torch.zeros(B, self.decoder_pme.attn_dim, device=device)
            context_h_to_p = torch.zeros(B, self.decoder_hme.attn_dim, device=device)

        # 🔁 디코딩 (teacher forcing)
        outputs_pme = []
        outputs_hme = []

        # 🔁 PME 디코딩
        prev_token_pme = tgt_seq_pme[:, 0]
        for t in range(1, tgt_seq_pme.size(1)):
            logits_pme, hidden_pme = self.decoder_pme(prev_token_pme, hidden_pme, context_p_to_h)
            outputs_pme.append(logits_pme.unsqueeze(1))
            use_tf = torch.rand(1).item() < teacher_forcing_ratio
            if t < tgt_seq_pme.size(1) - 1:
                prev_token_pme = tgt_seq_pme[:, t] if use_tf else logits_pme.argmax(dim=-1)

        # 🔁 HME 디코딩
        if not pme_only:
            prev_token_hme = tgt_seq_hme[:, 0]
            for t in range(1, tgt_seq_hme.size(1)):
                logits_hme, hidden_hme = self.decoder_hme(prev_token_hme, hidden_hme, context_h_to_p)
                outputs_hme.append(logits_hme.unsqueeze(1))
                use_tf = torch.rand(1).item() < teacher_forcing_ratio
                if t < tgt_seq_hme.size(1) - 1:
                    prev_token_hme = tgt_seq_hme[:, t] if use_tf else logits_hme.argmax(dim=-1)

        logits_pme = torch.cat(outputs_pme, dim=1)  # (B, T-1, vocab_size)

        if not pme_only:
            logits_hme = torch.cat(outputs_hme, dim=1)
        else:
            logits_hme = None
            context_h_to_p = None

        return logits_pme, logits_hme, context_p_to_h, context_h_to_p