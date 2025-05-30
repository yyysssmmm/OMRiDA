# 성능확인을 위해 만들어둔 dummy 내용임. 조원과 코드 통합하게 되면 업데이트 될 예정 





# models/attention.py

import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, enc_dim, hidden_dim):
        super().__init__()
        # ⚠️ 실제 구현 전까지는 빈 레이어로!
        self.dummy_context = nn.Parameter(torch.randn(enc_dim))

    def forward(self, features, hidden):
        """
        Return dummy context vector
        features: (B, C, H', W')
        hidden: (1, B, H)

        Returns:
            context: (B, C)
        """
        B = features.size(0)
        return self.dummy_context.unsqueeze(0).expand(B, -1)
