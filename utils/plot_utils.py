import matplotlib.pyplot as plt
import torch

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

def visualize_prediction(image, pred_str, target_str, figsize=(10, 3)):
    """수식 이미지와 예측/정답 문자열 시각화"""
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()
    
    plt.figure(figsize=figsize)
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title(f"Pred: {pred_str}\nTarget: {target_str}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def plot_learning_curve(losses, title="Training Loss"):
    plt.plot(losses, label="loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid()
    plt.legend()
    plt.show()
