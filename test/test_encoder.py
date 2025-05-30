import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))  # 상위 폴더 OMRiDA를 경로에 추가

import torch
from models.encoder import DenseNetEncoder

model = DenseNetEncoder(out_channels=512)

dummy = torch.randn(2, 3, 128, 512)  # 예: PME 입력
output = model(dummy)

print("Output shape:", output.shape)  # 예상: (2, 512, H', W')
