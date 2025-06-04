import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))  # 상위 폴더 OMRiDA를 경로에 추가

import torch
from models.attention import Attention

# 하이퍼파라미터 설정
B, C, H, W = 2, 512, 8, 32  # 예시 feature 맵 크기
hidden_dim = 256
attention_dim = 128

# Attention 모델 초기화
model = Attention(feature_dim=C, hidden_dim=hidden_dim, attention_dim=attention_dim)

# 더미 입력 생성
features = torch.randn(B, C, H, W)               # CNN 출력
hidden = torch.randn(B, hidden_dim)              # 디코더 hidden state
coverage = torch.zeros(B, H, W)                  # 초기 coverage

# forward 테스트
context, attn_map, coverage_out = model(features, hidden, coverage)

# 출력 확인
print("Context vector shape:", context.shape)         # 예상: (B, C)
print("Attention map shape:", attn_map.shape)         # 예상: (B, H, W)
print("Updated coverage shape:", coverage_out.shape)  # 예상: (B, H, W)
