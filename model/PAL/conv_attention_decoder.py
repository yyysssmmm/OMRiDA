import torch
import torch.nn as nn
import torch.nn.functional as F

# Convolutional Attention Block
class ConvAttentionBlock(nn.Module):
    def __init__(self, dim, dropout=0.5):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim * 2, kernel_size=3, padding=1)
        self.glu = nn.GLU(dim=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # x: (B, T, D)
        x = x.transpose(1, 2)           # (B, D, T)
        x = self.conv(x)                # (B, 2D, T)
        x = self.dropout(x)
        x = self.glu(x)                 # (B, D, T)
        x = x.transpose(1, 2)           # (B, T, D)
        return x

# Decoder
class PALDecoder(nn.Module):
    def __init__(self, d_model=256, vocab_size=1000, max_T=1024, dropout=0.5):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.pos_embed = nn.Parameter(torch.randn(1, max_T, d_model))  # learnable positional embedding

        # 3 Conv-Attention blocks
        self.attn_blocks = nn.Sequential(
            ConvAttentionBlock(d_model, dropout),
            ConvAttentionBlock(d_model, dropout),
            ConvAttentionBlock(d_model, dropout)
        )

        # MLP decoder head
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(256, vocab_size),
            nn.Dropout(dropout)
            # nn.Softmax(dim=1) # loss에서 처리해서 생략 가능 
        )

    def forward(self, encoder_out):
        """
        encoder_out: Tensor of shape (B, T, D) — encoder output
        Returns: (B, T, vocab_size)
        """
        B, T, D = encoder_out.shape
        x = encoder_out + self.pos_embed[:, :T, :]   # add positional embedding
        x = self.attn_blocks(x)                      # conv-attention blocks
        logits = self.mlp(x)                         # (B, T, V)
        return logits

# Test
encoder_out = torch.randn(2, 256, 256)
decoder = PALDecoder(d_model=256, vocab_size=30522)
logits = decoder(encoder_out)
print(logits.shape)