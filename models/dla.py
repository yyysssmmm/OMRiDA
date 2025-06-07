import torch
import torch.nn as nn
from .encoder import DenseNetEncoder
from .decoder import Decoder
from .DualCrossAttention import DualCrossAttention 

class DLAModel(nn.Module):
    """
    Dual-Cross-Attention 기반 MER 모델.
    - PME 이미지와 HME 이미지를 각각 인코딩
    - 양방향 Cross Attention 수행
    - PME 디코더는 HME context 사용, HME 디코더는 PME context 사용
    """

    def __init__(self, vocab_size, model_config=None):  # 🔧 model_config 추가
        super().__init__()

        # 🔧 config에서 받아온 파라미터 추출 (기본값 포함)
        emb_dim = model_config.get("emb_dim", 64)
        enc_dim = model_config.get("encoder_out_channels", 512)
        attn_dim = model_config.get("attention_dim", 256)
        hidden_dim = model_config.get("decoder_hidden_dim", 256)

        # 🔧 Encoder, Attention, Decoder 구성
        self.encoder_pme = DenseNetEncoder(out_channels=enc_dim)
        self.encoder_hme = DenseNetEncoder(out_channels=enc_dim)

        self.cross_attention = DualCrossAttention(
            pme_dim=enc_dim, hme_dim=enc_dim, attn_dim=attn_dim
        )

        self.decoder_pme = Decoder(vocab_size, emb_dim, attn_dim, hidden_dim)
        self.decoder_hme = Decoder(vocab_size, emb_dim, attn_dim, hidden_dim)

        self.hidden_dim = hidden_dim  # 디코더 초기 히든에 사용


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
        hidden_pme = torch.zeros(1, B, self.hidden_dim, device=device)  # 0으로 초기화하는게 최선일까?
        hidden_hme = torch.zeros(1, B, self.hidden_dim, device=device)  # 0으로 초기화하는게 최선일까?

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
        prev_token_pme = tgt_seq_pme[:, 0]  # <SOS>
        for t in range(1, tgt_seq_pme.size(1)):
            logits_pme, hidden_pme = self.decoder_pme(prev_token_pme, hidden_pme, context_p_to_h)
            outputs_pme.append(logits_pme.unsqueeze(1)) # (B, 1, V)
            use_tf = torch.rand(1).item() < teacher_forcing_ratio
            if use_tf:
                prev_token_pme = tgt_seq_pme[:, t]
            else:
                prev_token_pme = logits_pme.argmax(dim=-1)

        # 🔁 HME 디코딩
        if not pme_only:
            prev_token_hme = tgt_seq_hme[:, 0]  # <SOS>
            for t in range(1, tgt_seq_hme.size(1)):
                logits_hme, hidden_hme = self.decoder_hme(prev_token_hme, hidden_hme, context_h_to_p)
                outputs_hme.append(logits_hme.unsqueeze(1))
                use_tf = torch.rand(1).item() < teacher_forcing_ratio
                if use_tf:
                    prev_token_hme = tgt_seq_hme[:, t]
                else:
                    prev_token_hme = logits_hme.argmax(dim=-1)

        logits_pme = torch.cat(outputs_pme, dim=1)  # (B, T, vocab_size)

        if not pme_only:
            logits_hme = torch.cat(outputs_hme, dim=1)
        else:
            logits_hme = None
            context_h_to_p = None

        return logits_pme, logits_hme, context_p_to_h, context_h_to_p


    def predict_hme(self, images_hme, sos_idx, eos_idx, max_len=150):
        """
        HME 이미지를 기반으로 greedy decoding을 수행하는 추론용 메서드.

        Args:
            images_hme: torch.Tensor (B, 3, H, W) - HME 이미지
            max_len: int - 디코딩 최대 길이
            sos_idx: int - 시작 토큰 ID (<sos>)
            eos_idx: int - 종료 토큰 ID (<eos>)

        Returns:
            List[List[int]] - 배치 내 각 샘플별 예측 토큰 ID 시퀀스
        """
        self.eval()
        B = images_hme.size(0)
        device = images_hme.device
        hidden = torch.zeros(1, B, self.hidden_dim, device=device)

        with torch.no_grad():
            # HME 인코딩
            feat_hme = self.encoder_hme(images_hme)  # (B, C, H', W')

            # Dummy PME feature → CrossAttention 입력용 (내용은 상관 없음)
            dummy_pme = torch.zeros_like(feat_hme)
            _, context_h_to_p, _, _ = self.cross_attention(dummy_pme, feat_hme)  # (B, D)

            # 디코딩 시작
            preds = torch.full((B, 1), sos_idx, dtype=torch.long, device=device)  # 초기 <sos> 토큰

            for _ in range(max_len):
                last_token = preds[:, -1]  # 가장 마지막 토큰
                logits, hidden = self.decoder_hme(last_token, hidden, context_h_to_p)  # (B, vocab_size)
                next_token = logits.argmax(dim=-1).unsqueeze(1)  # greedy decoding
                preds = torch.cat([preds, next_token], dim=1)  # 누적

                if (next_token == eos_idx).all():  # 전체 배치에서 <eos> 나오면 중단
                    break

            return preds.tolist()


    def predict_pme(self, images_pme, sos_idx, eos_idx, max_len=150):
        """
        PME 이미지를 기반으로 greedy decoding을 수행하는 추론용 메서드.

        Args:
            images_pme: torch.Tensor (B, 3, H, W) - PME 이미지
            max_len: int - 디코딩 최대 길이
            sos_idx: int - 시작 토큰 ID (<sos>)
            eos_idx: int - 종료 토큰 ID (<eos>)

        Returns:
            List[List[int]] - 배치 내 각 샘플별 예측 토큰 ID 시퀀스
        """
        self.eval()
        B = images_pme.size(0)
        device = images_pme.device
        hidden = torch.zeros(1, B, self.hidden_dim, device=device)

        with torch.no_grad():
            feat_pme = self.encoder_pme(images_pme)
            dummy_hme = torch.zeros_like(feat_pme)
            context_p_to_h, _, _, _ = self.cross_attention(feat_pme, dummy_hme)

            preds = torch.full((B, 1), sos_idx, dtype=torch.long, device=device)

            for _ in range(max_len):
                last_token = preds[:, -1]
                logits, hidden = self.decoder_pme(last_token, hidden, context_p_to_h)
                next_token = logits.argmax(dim=-1).unsqueeze(1)
                preds = torch.cat([preds, next_token], dim=1)

                if (next_token == eos_idx).all():
                    break

            return preds.tolist()

