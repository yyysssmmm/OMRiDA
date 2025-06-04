import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))  # 상위 폴더 OMRiDA 경로에 추가

import torch
from models.context_matcher import ContextMatcher

# 더미 입력 생성
B, T, D = 4, 10, 256  # 배치 크기 4, 시퀀스 길이 10, 차원 256
context_h = torch.randn(B, T, D)        # 수기 수식 context
context_p = torch.randn(B, T, D)        # 인쇄 수식 context

# ContextMatcher 초기화
matcher = ContextMatcher()

# forward 테스트
loss = matcher(context_h, context_p)

# 출력 확인
print("Context Matching Loss:", loss.item())
