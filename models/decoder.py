import torch
import torch.nn as nn

class Decoder(nn.Module):
    """
    Decoder module for HMER.
    
    Inputs:
    - prev_token: previous token (B,)
    - hidden: previous hidden state (1, B, H)
    - context: attention-based context vector (B, enc_dim), currently dummy in test
    
    Outputs:
    - logits: unnormalized scores over vocabulary (B, vocab_size)
    - hidden: updated hidden state (1, B, H)
    """
    def __init__(self, vocab_size, emb_dim, enc_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.gru = nn.GRU(emb_dim + enc_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, prev_token, hidden, context):
        """
        prev_token: (B,)             # 이전 토큰 (정수 인덱스)
        hidden: (1, B, H)            # 이전 hidden state
        context: (B, enc_dim)        # attention context
        """
        embedded = self.embed(prev_token).unsqueeze(1)      # (B, 1, emb_dim)
        context = context.unsqueeze(1)                      # (B, 1, enc_dim)
        gru_input = torch.cat([embedded, context], dim=2)   # (B, 1, emb+enc)
        output, hidden = self.gru(gru_input, hidden)        # output: (B, 1, H)
        output_logits = self.fc_out(output.squeeze(1))      # (B, vocab_size)
        return output_logits, hidden
