import torch
import torch.nn as nn

class Decoder(nn.Module):
    """
    Decoder module for MER.

    Inputs:
    - prev_token: previous token (B,)
    - hidden: previous hidden state (1, B, H)
    - context: attention-based context vector (B, attn_dim) if use_context=True, else None

    Outputs:
    - logits: unnormalized scores over vocabulary (B, vocab_size)
    - hidden: updated hidden state (1, B, H)
    """
    def __init__(self, vocab_size, emb_dim, attn_dim, hidden_dim, use_context=True, num_layers=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_context = use_context
        self.embed = nn.Embedding(vocab_size, emb_dim)
        
        if use_context:
            self.attn_dim = attn_dim
            self.gru = nn.GRU(emb_dim + attn_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        else:
            self.gru = nn.GRU(emb_dim, hidden_dim, num_layers=num_layers, batch_first=True)

        self.fc_out = nn.Linear(hidden_dim, vocab_size)

        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.embed.weight, -0.1, 0.1)

        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)

    def forward(self, prev_token, hidden, context=None):
        embedded = self.embed(prev_token).unsqueeze(1)  # (B, 1, emb_dim)

        if self.use_context:
            if context is None:
                raise ValueError("Context must be provided when use_context=True")
            context = context.unsqueeze(1)  # (B, 1, attn_dim)
            gru_input = torch.cat([embedded, context], dim=2)  # (B, 1, emb + attn)
        else:
            gru_input = embedded  # (B, 1, emb_dim)

        output, hidden = self.gru(gru_input, hidden)  # output: (B, 1, H)
        output_logits = self.fc_out(output.squeeze(1))  # (B, vocab_size)

        return output_logits, hidden
