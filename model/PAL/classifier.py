import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, d_model=256, vocab_size=30522, dropout=0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, vocab_size)
            # nn.Softmax(dim=1) # loss에서 처리해서 생략 가능 
        )

    def forward(self, x):
        """
        x: (B, T, D) — encoder feature sequence
        returns: (B, T, V) — class logits per token
        """
        B, T, D = x.shape
        x = self.classifier(x.view(B * T, D))  # (B*T, V)
        return x.view(B, T, -1)  # (B, T, V)
