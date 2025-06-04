import matplotlib.pyplot as plt
import torch
import numpy as np


def decode_sequence(token_ids, vocab):
    """
    토큰 인덱스 시퀀스를 수식 문자열로 디코딩.
    
    Args:
        token_ids: (B, T) 텐서
        vocab: Vocab 객체 (idx2token 포함)
    
    Returns:
        List of strings
    """
    results = []
    for seq in token_ids:
        tokens = []
        for idx in seq:
            token = vocab.idx2token[int(idx)]
            if token == "<pad>":
                continue
            if token == "<eos>":
                break
            tokens.append(token)
        results.append(" ".join(tokens))
    return results


def token_accuracy(preds, targets, pad_token=0):
    """
    token 단위 정확도 계산 (padding 무시)
    
    Args:
        preds: (B, T) 예측 시퀀스
        targets: (B, T) 정답 시퀀스
    
    Returns:
        accuracy: float
    """
    mask = targets != pad_token
    correct = (preds == targets) & mask
    accuracy = correct.sum().item() / mask.sum().item()
    return accuracy


def visualize_prediction(image, pred_str, target_str, figsize=(10, 3)):
    """
    수식 이미지와 예측/정답 문자열을 시각화
    
    Args:
        image: (3, H, W) torch.Tensor or (H, W, 3) numpy array
        pred_str: 예측 문자열
        target_str: 정답 문자열
    """
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
    
    plt.figure(figsize=figsize)
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title(f"Pred: {pred_str}\nTarget: {target_str}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def plot_learning_curve(losses, title="Training Loss"):
    """
    학습 손실 시각화
    
    Args:
        losses: List of floats
    """
    plt.plot(losses, label="loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid()
    plt.legend()
    plt.show()
