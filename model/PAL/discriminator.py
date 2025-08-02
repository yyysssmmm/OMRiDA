import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.linear1 = nn.Linear(d_model, 512)
        self.dropout = nn.Dropout(0.2)
        self.linear2 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        x: (B, T, D) — feature sequence from encoder or decoder
        returns: (B, T) — domain prediction probabilities
        """

        # dimension이 (B, H, W, D)인지 (B, T, D)인지 모름
        # 논문 수식 기준 (B, T, D)로 일단 구성
        if x.dim() == 3:
            B, T, D = x.shape
            x = x.view(B * T, D)

        x = self.linear1(x)             # (B, T, 512)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)             # (B, T, 1)
        x = self.sigmoid(x)             # (B, T, 1)
        return x.squeeze(B, T)            # (B, T)