import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.d_model = d_model
        self.linear1 = nn.Linear(d_model, 512)
        self.dropout = nn.Dropout(0.2)
        self.linear2 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def _to_seq(self, x):
        """
        입력을 (B, T, D)로 변환:
        - (B, T, D) 그대로
        - (B, D, H, W) -> (B, H*W, D)
        - (B, H, W, D) -> (B, H*W, D)
        """
        if x.dim() == 3:
            return x  # (B, T, D)

        if x.dim() == 4:
            B = x.size(0)
            if x.size(1) == self.d_model:
                x = x.permute(0, 2, 3, 1).contiguous()
            B, H, W, D = x.shape
            x = x.view(B, H * W, D)
            return x

        raise ValueError(f"Unsupported input shape {tuple(x.shape)}")

    def forward(self, x):
        """
        x: (B, T, D) 또는 (B, D, H, W)/(B, H, W, D)
        out: (B, T)
        """
        x = self._to_seq(x)         # (B, T, D)
        B, T, D = x.shape
        x = x.reshape(B * T, D)     # (B*T, D)

        x = F.relu(self.linear1(x), inplace=True)
        x = self.dropout(x)
        x = self.linear2(x)         # (B*T, 1)
        x = self.sigmoid(x)         # (B*T, 1)

        x = x.view(B, T)            # (B, T)
        return x