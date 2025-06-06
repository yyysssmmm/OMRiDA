import torch
import torch.nn as nn

from .encoder import DenseNetEncoder
from .decoder import Decoder
from .DualCrossAttention import DualCrossAttention

class DLAModel(nn.Module):
    def __init__(self, vocab_size, model_config=None, is_paired=True):
        super().__init__()

        emb_dim = model_config.get("emb_dim", 64)
        enc_dim = model_config.get("encoder_out_channels", 512)
        attn_dim = model_config.get("attention_dim", 256)
        hidden_dim = model_config.get("decoder_hidden_dim", 256)

        self.is_paired = is_paired
        self.hidden_dim = hidden_dim

        # Encoder
        self.encoder_pme = DenseNetEncoder(out_channels=enc_dim)
        self.init_hidden_proj_pme = nn.Linear(enc_dim, hidden_dim)

        if is_paired:
            self.encoder_hme = DenseNetEncoder(out_channels=enc_dim)
            self.init_hidden_proj_hme = nn.Linear(enc_dim, hidden_dim)
            self.cross_attention = DualCrossAttention(
                pme_dim=enc_dim, hme_dim=enc_dim, attn_dim=attn_dim
            )

        # Decoder
        if is_paired:
            self.decoder_hme = Decoder(vocab_size, emb_dim, attn_dim, hidden_dim, use_context=True)
            self.decoder_pme = Decoder(vocab_size, emb_dim, attn_dim, hidden_dim, use_context=True)
        else:
            self.decoder_pme = Decoder(vocab_size, emb_dim, attn_dim=None, hidden_dim=hidden_dim, use_context=False)

    def forward(self, images_pme, images_hme=None, tgt_seq_pme=None, tgt_seq_hme=None, teacher_forcing_ratio=0.5):
        B, T = tgt_seq_pme.shape
        device = images_pme.device

        feat_pme = self.encoder_pme(images_pme)
        hidden_pme = self.init_hidden_proj_pme(feat_pme.mean([2, 3])).unsqueeze(0)  # (1, B, H) (Global Average Pooling)

        if self.is_paired:
            feat_hme = self.encoder_hme(images_hme)
            hidden_hme = self.init_hidden_proj_hme(feat_hme.mean([2, 3])).unsqueeze(0)  # GAP 

            context_p_to_h, context_h_to_p, _, _ = self.cross_attention(feat_pme, feat_hme)
        else:
            context_p_to_h = None
            context_h_to_p = None
            hidden_hme = None

        outputs_pme = []
        outputs_hme = []

        prev_token_pme = tgt_seq_pme[:, 0]
        for t in range(1, tgt_seq_pme.size(1)):
            logits_pme, hidden_pme = self.decoder_pme(prev_token_pme, hidden_pme, context_p_to_h)
            outputs_pme.append(logits_pme.unsqueeze(1))
            use_tf = torch.rand(1).item() < teacher_forcing_ratio
            prev_token_pme = tgt_seq_pme[:, t] if use_tf else logits_pme.argmax(dim=-1)

        if self.is_paired:
            prev_token_hme = tgt_seq_hme[:, 0]
            for t in range(1, tgt_seq_hme.size(1)):
                logits_hme, hidden_hme = self.decoder_hme(prev_token_hme, hidden_hme, context_h_to_p)
                outputs_hme.append(logits_hme.unsqueeze(1))
                use_tf = torch.rand(1).item() < teacher_forcing_ratio
                prev_token_hme = tgt_seq_hme[:, t] if use_tf else logits_hme.argmax(dim=-1)

        logits_pme = torch.cat(outputs_pme, dim=1)
        logits_hme = torch.cat(outputs_hme, dim=1) if self.is_paired else None

        return logits_pme, logits_hme, context_p_to_h, context_h_to_p

    # greedy
    def predict(self, image, max_len=150, sos_idx=1, eos_idx=2):
        self.eval()
        device = image.device

        if image.dim() == 3:  # (C, H, W) → 단일 이미지인 경우 (1, C, H, W)로 변환
            image = image.unsqueeze(0)

        batch_size = image.size(0)
        preds = []

        with torch.no_grad():
            feat_pme = self.encoder_pme(image)
            hidden = self.init_hidden_proj_pme(feat_pme.mean([2, 3])).unsqueeze(0)  # (1, B, H)

            if self.is_paired:
                feat_hme = self.encoder_hme(image)
                context, _, _, _ = self.cross_attention(feat_pme, feat_hme)
            else:
                context = None

            decoder = self.decoder_pme
            cur_tokens = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=device)  # (B, 1)

            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

            for _ in range(max_len):
                last_token = cur_tokens[:, -1]  # (B,)
                logits, hidden = decoder(last_token, hidden, context)
                next_token = logits.argmax(dim=-1, keepdim=True)  # (B, 1)
                cur_tokens = torch.cat([cur_tokens, next_token], dim=1)  # (B, T+1)

                # EOS 처리
                finished = finished | (next_token.squeeze(1) == eos_idx)
                if finished.all():
                    break

            pred_list = cur_tokens[:, 1:].tolist()  # [B, T] → sos 제외
            return pred_list

