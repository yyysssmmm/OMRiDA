import torch
import torch.nn as nn

from .encoder import DenseNetEncoder
from .decoder import Decoder
from .DualCrossAttention import DualCrossAttention


class DLAModel(nn.Module):
    def __init__(self, vocab_size, model_config=None):
        super().__init__()

        # 🔧 설정 파라미터 추출
        emb_dim = model_config.get("emb_dim", 64)
        enc_dim = model_config.get("encoder_out_channels", 512)
        attn_dim = model_config.get("attention_dim", 256)
        hidden_dim = model_config.get("decoder_hidden_dim", 256)

        self.hidden_dim = hidden_dim
        self.attn_dim = attn_dim

        # ✅ 인코더 (PME, HME 각각)
        self.encoder_pme = DenseNetEncoder(out_channels=enc_dim)
        self.encoder_hme = DenseNetEncoder(out_channels=enc_dim)

        # ✅ 초기 히든 state 생성용 projection layer
        self.init_hidden_proj_pme = nn.Linear(enc_dim, hidden_dim)
        self.init_hidden_proj_hme = nn.Linear(enc_dim, hidden_dim)

        # ✅ context + hidden → hidden_dim projection
        self.context_proj = nn.Linear(hidden_dim + attn_dim, hidden_dim)

        # ✅ Cross Attention
        self.cross_attention = DualCrossAttention(
            pme_dim=enc_dim, hme_dim=enc_dim, attn_dim=attn_dim
        )

        # ✅ 디코더 (PME, HME 동일 구조 → 공유)
        self.decoder = Decoder(vocab_size, emb_dim, hidden_dim)

    def forward(self, images_pme, images_hme=None, tgt_seq_pme=None, tgt_seq_hme=None, teacher_forcing_ratio=0.5):
        B, T = tgt_seq_pme.shape
        device = images_pme.device

        # 🔹 PME 인코딩 + GAP → 초기 hidden
        feat_pme = self.encoder_pme(images_pme)
        hidden_pme = self.init_hidden_proj_pme(feat_pme.mean([2, 3])).unsqueeze(0)  # (1, B, H)

        # 🔹 HME가 주어진 경우 → paired training
        if images_hme is not None:
            feat_hme = self.encoder_hme(images_hme)
            hidden_hme = self.init_hidden_proj_hme(feat_hme.mean([2, 3])).unsqueeze(0)

            # 🔹 양방향 Cross Attention
            context_p_to_h, context_h_to_p, _, _ = self.cross_attention(feat_pme, feat_hme)

            # 🔹 context + hidden → projected hidden
            hidden_pme = self.context_proj(torch.cat([hidden_pme.squeeze(0), context_p_to_h], dim=-1)).unsqueeze(0)
            hidden_hme = self.context_proj(torch.cat([hidden_hme.squeeze(0), context_h_to_p], dim=-1)).unsqueeze(0)
        else:
            context_p_to_h = context_h_to_p = hidden_hme = None

        # 🔹 PME 디코딩
        outputs_pme = []
        prev_token_pme = tgt_seq_pme[:, 0]  # (B,) - <sos>
        for t in range(1, tgt_seq_pme.size(1)):
            logits_pme, hidden_pme = self.decoder(prev_token_pme, hidden_pme)
            outputs_pme.append(logits_pme.unsqueeze(1))
            use_tf = torch.rand(1).item() < teacher_forcing_ratio
            prev_token_pme = tgt_seq_pme[:, t] if use_tf else logits_pme.argmax(dim=-1)

        # 🔹 HME 디코딩 (paired인 경우만)
        outputs_hme = []
        if images_hme is not None:
            prev_token_hme = tgt_seq_hme[:, 0]  # <sos>
            for t in range(1, tgt_seq_hme.size(1)):
                logits_hme, hidden_hme = self.decoder(prev_token_hme, hidden_hme)
                outputs_hme.append(logits_hme.unsqueeze(1))
                use_tf = torch.rand(1).item() < teacher_forcing_ratio
                prev_token_hme = tgt_seq_hme[:, t] if use_tf else logits_hme.argmax(dim=-1)

        # 🔹 최종 로그잇 반환
        logits_pme = torch.cat(outputs_pme, dim=1)
        logits_hme = torch.cat(outputs_hme, dim=1) if outputs_hme else None

        return logits_pme, logits_hme, context_p_to_h, context_h_to_p

    def predict(self, image, image_hme=None, max_len=150, sos_idx=1, eos_idx=2):
        self.eval()
        device = image.device

        if image.dim() == 3:  # (C, H, W) 단일 이미지를 배치 형태로
            image = image.unsqueeze(0)

        batch_size = image.size(0)

        with torch.no_grad():
            # 🔹 PME 인코딩 + GAP → 초기 hidden
            feat_pme = self.encoder_pme(image)
            hidden = self.init_hidden_proj_pme(feat_pme.mean([2, 3])).unsqueeze(0)

            # 🔹 HME가 주어진 경우 → context 반영
            if image_hme is not None:
                feat_hme = self.encoder_hme(image_hme)
                context, _, _, _ = self.cross_attention(feat_pme, feat_hme)
                hidden = self.context_proj(torch.cat([hidden.squeeze(0), context], dim=-1)).unsqueeze(0)

            # 🔹 토큰 생성 루프
            cur_tokens = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=device)
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

            for _ in range(max_len):
                last_token = cur_tokens[:, -1]
                logits, hidden = self.decoder(last_token, hidden)
                next_token = logits.argmax(dim=-1, keepdim=True)
                cur_tokens = torch.cat([cur_tokens, next_token], dim=1)
                finished = finished | (next_token.squeeze(1) == eos_idx)
                if finished.all():
                    break

            pred_list = cur_tokens[:, 1:].tolist()  # <sos> 제외
            return pred_list
