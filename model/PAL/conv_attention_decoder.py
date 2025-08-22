import torch
import torch.nn as nn
import torch.nn.functional as F

# Conv-256 -> GLU -> Dropout 0.5: (B, D, L) -> (B, D, L)
class ConvAttnLayer(nn.Module): 
    def __init__(self, d_model: int, kernel_size: int = 3, dropout: float = 0.5):
        super().__init__()
        self.pad_left = kernel_size - 1
        self.conv = nn.Conv1d(d_model, 2 * d_model, kernel_size, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):                  # x: (B, D, L)
        x = F.pad(x, (self.pad_left, 0))   # causal pad (left only)
        y = self.conv(x)                   # (B, 2D, L)
        a, b = y.chunk(2, dim=1)           # GLU split (채널 축으로 갈라서 A를 B로 원소별 곱하기)
        z = a * torch.sigmoid(b)           # (B, D, L)
        return self.dropout(z)
    
# 3 ConvAttnLayer: (B, D, L) -> (B, D, L)
class ConvAttnBlock(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.5):
        super().__init__()
        self.layers = nn.ModuleList([ConvAttnLayer(d_model, 3, dropout) for _ in range(3)])

    def forward(self, x):
        residual = x
        for layer in self.layers:
            x = layer(x)
        return x + residual
    
class PALDecoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, max_enc_T: int = 8192):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        # encoder-side absolute position embedding
        # 인코더 시퀀스 길이 N에 대해 (1, N, D)로 잘라서 더함
        self.enc_pos = nn.Parameter(torch.randn(1, max_enc_T, d_model))

        # ConvAttnBlock 3개 쌓아서 stack 만들기
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.emb_drop = nn.Dropout(0.5)
        self.conv_blocks = nn.ModuleList([ConvAttnBlock(d_model) for _ in range(3)])

        # block별 상태 요약 state projection
        self.Ws = nn.ModuleList([nn.Linear(d_model, d_model, bias=True) for _ in range(3)])

        # output head: 256 units -> Dropout 0.5 -> vocab
        self.mlp = nn.Sequential(
            nn.Linear(2 * d_model, 256),
            nn.Dropout(0.5),
            nn.Linear(256, vocab_size),
            nn.Dropout(0.5)
        )

    def forward(self, encoder_out, tgt_input_ids):
        """
        encoder_out: (B, N, D) -> 인코더 2D feature을 열 기준으로 펼친 시퀀스
        tgt_input_ids: (B, L) -> teacher forcing용 입력 (토큰 prefix)
        returns: (B, L, V) -> 각 타임스텝의 logits
        """
        B, N, D = encoder_out.shape
        assert D == self.d_model, f"D mismatch: enc={D}, dec={self.d_model}"

        # encoder location embedding 추가
        e = encoder_out + self.enc_pos[:, :N, :]             # (B, N, D)

        # 토큰 embedding -> ConvAttnBlock으로 보낼 초기 hidden 준비
        dec = self.emb_drop(self.tok_embed(tgt_input_ids))   # (B, L, D)
        h = dec.transpose(1, 2).contiguous()                 # (B, D, L)

        # 마지막 block 컨텍스트 모아두기 (출력 head 입력용)
        C_last_list = None

        # multi-hop: ConvAttnBlock 순서대로 진행
        for l in range(3):
            # residual connection
            h_block_in = h                                      # (B, D, L)
            # ConvAttnBlock
            h = self.conv_blocks[l](h)                          # (B, D, L)
            # 각 시점 t의 hidden
            h_seq = h.transpose(1, 2).contiguous()              # (B, L, D)
            # block별 상태 state projection
            s = self.Ws[l](h_seq)                               # (B, L, D)

            # step residual connection: 각 시점 t에서 dot-product attention -> hidden에 추가
            save_c = (l == 2)
            if save_c:
                C_last_list = []

            for t in range(h_seq.size(1)):
                s_t = s[:, t, :]                                      # (B, D)
                # dot-product energy
                energy = torch.einsum('bnd,bd->bn', e, s_t)           # (B, N)
                alpha_t = F.softmax(energy, dim=-1)                   # (B, N)

                # 컨텍스트
                c_t = torch.bmm(alpha_t.unsqueeze(1), e).squeeze(1)   # (B, D)

                # step residual connection
                h_seq[:, t, :] = h_seq[:, t, :] + c_t

                # 마지막 block -> 컨텍스트 저장
                if save_c:
                    C_last_list.append(c_t.unsqueeze(1))        # (B, 1, D)

            # 시퀀스 (B, D, L) -> block residual connection
            h = h_seq.transpose(1, 2).contiguous()              # (B, D, L)
            h = h + h_block_in 

        # 마지막 block 컨텍스트 시퀀스, 최종 hidden 시퀀스
        C_last = torch.cat(C_last_list, dim=1)                  # (B, L, D)
        H_last = h.transpose(1, 2).contiguous()                 # (B, L, D)
        mlp_in = torch.cat([C_last, H_last], dim=-1)            # (B, L, 2D)
        logits = self.mlp(mlp_in)                               # (B, L, V)

        return logits