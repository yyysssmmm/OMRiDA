import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    """재현 가능한 실험을 위한 시드 고정."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"🌱 Seed set to {seed}")
