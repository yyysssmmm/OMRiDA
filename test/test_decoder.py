import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))  # 상위 폴더 OMRiDA를 경로에 추가

import torch
from models.decoder import Decoder

# 하이퍼파라미터 정의
B = 2  # batch size
vocab_size = 100
emb_dim = 64
enc_dim = 512
hidden_dim = 256

# Decoder 모델 인스턴스 생성
decoder = Decoder(vocab_size=vocab_size, emb_dim=emb_dim, enc_dim=enc_dim, hidden_dim=hidden_dim)

# 더미 입력 생성
prev_token = torch.randint(0, vocab_size, (B,))       # (B,)
hidden = torch.zeros(1, B, hidden_dim)                # (1, B, H)
context = torch.randn(B, enc_dim)                     # (B, enc_dim)

# Forward 연산
logits, next_hidden = decoder(prev_token, hidden, context)

# 출력 확인
print("Logits shape:", logits.shape)           # (B, vocab_size)
print("Next hidden shape:", next_hidden.shape) # (1, B, hidden_dim)
