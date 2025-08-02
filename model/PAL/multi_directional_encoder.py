import torch
import torch.nn as nn

# CNN Block
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        layers = []
        for _ in range(4):
            layers += [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ]
            in_channels = out_channels
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return self.body(x)

# MD-Transition Layer
'''
4방향 LSTM 대신 4방향 Conv 사용해서 lightweight 대체
성능 향상에 비해서 메모리 사용량이 매우 크고 속도도 느림
MDLSTM을 다른 걸로 대체해보는 것도 실험할 수 있지 않을까?
MDLSTM official github 있어서 추후 MDLSTM으로 교체시 사용 가능
https://github.com/suhaspillai/HandwritingRecognition-with-MultiDimensionalRecurrentNeuralNetworks.git
'''
class MDTransitionLayer(nn.Module):
    def __init__(self, channels, mdlstm_out_channels, dropout=0.0):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv = nn.Conv2d(channels, mdlstm_out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.pool(x)
        out_ltr = self.conv(x)
        out_rtl = torch.flip(self.conv(torch.flip(x, dims=[3])), dims=[3])
        out_ttb = self.conv(x)
        out_btt = torch.flip(self.conv(torch.flip(x, dims=[2])), dims=[2])
        out = out_ltr + out_rtl + out_ttb + out_btt
        return self.dropout(out)

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
        x = self.md1(x)

        x = self.block2(x)
        x = self.md2(x)

        x = self.block3(x)
        x = self.md3(x)

        x = self.block4(x)
        x = self.md4(x)

        # CNN output: (B, C=256, H, W)
        x = x.permute(0, 2, 3, 1)         # (B, H, W, D)
        B, H, W, D = x.shape
        x = x.reshape(B, H * W, D)        # (B, T, D)

        return x  # Now: (B, T, D)

# Test
encoder = PALEncoder()
x = torch.randn(1, 1, 128, 512)
out = encoder(x)
print(out.shape)