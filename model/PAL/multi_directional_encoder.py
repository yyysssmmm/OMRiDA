import torch
import torch.nn as nn

# CNN Block: [3x3 conv, BN, ReLU] x4
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        layers = []
        c_in = in_channels
        for _ in range(4):
            layers += [
                nn.Conv2d(c_in, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ]
            c_in = out_channels
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return self.body(x)

# MD-Transition Layer: MaxPool -> 가로 BiLSTM + 세로 BiLSTM -> Dropout
# MD-LSTM 대신 행/열 BiLSTM 사용 -> 네 방향 컨텍스트 효과 동일
class MDTransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        self.h_lstm = nn.LSTM(
            input_size=in_channels, hidden_size=out_channels,
            num_layers=1, bidirectional=True, batch_first=True
        )
        self.v_lstm = nn.LSTM(
            input_size=in_channels, hidden_size=out_channels,
            num_layers=1, bidirectional=True, batch_first=True
        )
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.pool(x)
        B, C_in, H, W = x.shape

        # 가로 스캔: (B, C_in, H, W) -> (B, H, W, C_in) -> (B*H, W, C_in)
        h_in  = x.permute(0, 2, 3, 1).contiguous().view(B * H, W, C_in)
        # BiLSTM: (B*H, W, 2*out_channels)
        h_out, _ = self.h_lstm(h_in)
        # 양방향 분리 후 합: (B*H, W, out_channels)                                      
        h_fw, h_bw = torch.chunk(h_out, 2, dim=2)
        h_sum = (h_fw + h_bw).view(B, H, W, -1)                           

        # 세로 스캔: (B, C_in, H, W) -> (B, W, H, C_in) -> (B*W, H, C_in)
        v_in  = x.permute(0, 3, 2, 1).contiguous().view(B * W, H, C_in)
        # BiLSTM: (B*W, H, 2*out_channels)
        v_out, _ = self.v_lstm(v_in)
        # 양방향 분리 후 합: (B*W, H, out_channels) -> (B, H, W, out_channels)                                     
        v_fw, v_bw = torch.chunk(v_out, 2, dim=2)
        v_sum = (v_fw + v_bw).view(B, W, H, -1).permute(0, 2, 1, 3)   # 축 원래대로 바꾸기      

        out = (h_sum + v_sum).permute(0, 3, 1, 2).contiguous()            
        return self.drop(out)

# Encoder
class PALEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = CNNBlock(1, 32)
        self.md1 = MDTransitionLayer(32, 64, dropout=0.0)

        self.block2 = CNNBlock(64, 64)
        self.md2 = MDTransitionLayer(64, 64, dropout=0.2)

        self.block3 = CNNBlock(64, 64)
        self.md3 = MDTransitionLayer(64, 128, dropout=0.25)

        self.block4 = CNNBlock(128, 128)
        self.md4 = MDTransitionLayer(128, 256, dropout=0.35)

    def forward(self, x):
        x = self.block1(x)
        x = self.md1(x)     # C=64

        x = self.block2(x)
        x = self.md2(x)     # C=64

        x = self.block3(x)
        x = self.md3(x)     # C=128

        x = self.block4(x)
        x = self.md4(x)     # C=128, H/16, W/16

        return x            # (B, 256, H', W')