import torch
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights


class DenseNetEncoder(nn.Module):
    """
    DenseNet-based Encoder for offline HMER/PMER image inputs.
    
    This module extracts spatial feature maps from input images using DenseNet121.
    The final classifier layer is removed, and the convolutional backbone is used to 
    preserve spatial resolution.

    Inputs:
    - x: image tensor of shape (B, C=3, H=128, W=512) [normalized PME images]

    Outputs:
    - feature maps of shape (B, out_channels, H', W') where out_channels is typically 512

    Note:
    - The input size should maintain an appropriate aspect ratio (~1:4) for effective encoding.
    - Output spatial dimensions H', W' depend on DenseNet internal downsampling.
    """
    def __init__(self, out_channels=512):
        super().__init__()
        # densenet = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        densenet = densenet121(weights=None)    # 이번 프로젝트는 imageNet 데이터에 비교하면 수식글씨로, imageNet 가중치가 방해될 수도 있어 사전학습 가중치 없이 불러오는게 유리할수도 있음

        # Remove classification head
        features = list(densenet.features.children())

        # Use all layers except final norm+relu+avgpool
        self.backbone = nn.Sequential(*features[:-1])

        # Optional: project feature channels to desired out_channels
        self.project = nn.Conv2d(1024, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.backbone(x)            # Shape: (B, 1024, H', W')
        x = self.project(x)             # Shape: (B, out_channels, H', W')
        return x
