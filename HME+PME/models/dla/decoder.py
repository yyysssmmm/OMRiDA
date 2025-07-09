import torch
import torch.nn as nn

class Decoder(nn.Module):
    """
    Decoder module for MER.
    - context는 사용하지 않으며, 단순히 prev_token + hidden만으로 동작.
    - context를 사용할 경우, dla.py에서 hidden에 context를 concat하여 넘겨줄 것.

    Inputs:
    - prev_token: previous token (B,)
    - hidden: previous hidden state (1, B, H)

    Outputs:
    - logits: unnormalized scores over vocabulary (B, vocab_size)
    - hidden: updated hidden state (1, B, H)
    """
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(vocab_size, emb_dim)

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

    def forward(self, prev_token, hidden):
        embedded = self.embed(prev_token).unsqueeze(1)  # (B, 1, emb_dim)
        output, hidden = self.gru(embedded, hidden)     # (B, 1, H)
        output_logits = self.fc_out(output.squeeze(1))  # (B, vocab_size)
        return output_logits, hidden
