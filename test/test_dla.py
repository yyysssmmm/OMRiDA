import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))  # 상위 폴더 OMRiDA를 경로에 추가

import torch
from models.dla import DLAModel

if __name__ == "__main__":
    model = DLAModel(vocab_size=100)

    images = torch.randn(2, 3, 128, 512)
    tgt_seq = torch.randint(0, 100, (2, 20))  # (B, T)

    output = model(images, tgt_seq)
    print("Output shape:", output.shape)  # (2, 19, vocab_size)
