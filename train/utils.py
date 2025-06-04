import os
import random
import json
import torch
import numpy as np
import matplotlib.pyplot as plt


def set_seed(seed: int = 42):
    """재현 가능한 실험을 위한 시드 고정."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"🌱 Seed set to {seed}")


def save_log(log_dict, save_path="log.json"):
    """학습 로그를 JSON으로 저장."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(log_dict, f, indent=2, ensure_ascii=False)
    print(f"📄 로그 저장 완료: {save_path}")


def plot_loss_curve(loss_list, save_path=None):
    """Loss 곡선 시각화."""
    plt.figure(figsize=(8, 4))
    plt.plot(loss_list, marker='o', label="Train Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"📊 Loss curve saved to {save_path}")
    else:
        plt.show()
