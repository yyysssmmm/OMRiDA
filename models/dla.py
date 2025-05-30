import torch
import torch.nn as nn
from .encoder import DenseNetEncoder
from .attention import Attention
from .decoder import Decoder

class DLAModel(nn.Module):
    """
    DLA 논문 기반의 MER 모델.
    - Encoder: DenseNet 기반 시각 특징 추출기
    - Attention: context vector 생성
    - Decoder: GRU 기반 시퀀스 예측기
    """

    def __init__(self, vocab_size, emb_dim=64, enc_dim=512, hidden_dim=256):
        super().__init__()
        self.encoder = DenseNetEncoder(out_channels=enc_dim)
        self.attention = Attention(enc_dim, hidden_dim)
        self.decoder = Decoder(vocab_size, emb_dim, enc_dim, hidden_dim)

    def forward(self, images, tgt_seq, teacher_forcing_ratio=0.5):
        """
        Forward pass for training/inference

        Args:
            images: (B, 3, H, W) - input image
            tgt_seq: (B, T) - ground truth token ids
            teacher_forcing_ratio: float - ratio to use teacher forcing

        Returns:
            output: (B, T-1, vocab_size) - predicted logits
        """
        B, T = tgt_seq.shape
        device = images.device

        features = self.encoder(images)  # (B, C, H', W')
        hidden = torch.zeros(1, B, self.decoder.hidden_dim, device=device)

        outputs = []
        prev_token = tgt_seq[:, 0]  # <sos> token

        for t in range(1, T):
            context = self.attention(features, hidden)
            logits, hidden = self.decoder(prev_token, hidden, context)
            outputs.append(logits.unsqueeze(1))

            use_tf = torch.rand(1).item() < teacher_forcing_ratio
            prev_token = tgt_seq[:, t] if use_tf else logits.argmax(dim=1)

        return torch.cat(outputs, dim=1)  # (B, T-1, vocab_size)
