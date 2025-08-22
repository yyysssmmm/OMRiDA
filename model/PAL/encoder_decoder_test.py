import os
import cv2
import numpy as np
import torch

from multi_directional_encoder import PALEncoder
from conv_attention_decoder import PALDecoder

def imread_unicode(path, flags=cv2.IMREAD_GRAYSCALE):
    stream = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(stream, flags)

# 1) Encoder
encoder = PALEncoder().eval()

base_dir = r"C:/Users/sophi/OneDrive/바탕 화면/Folder/2025 HYU/인공지능프로젝트/data/crohme/dummy"
img_path  = os.path.join(base_dir, "hme_preprocessed")
image_path = os.path.normpath(os.path.join(img_path, "90_carlos.png"))
img = imread_unicode(image_path)
if img is None:
    raise FileNotFoundError(f"Image not found: {image_path}")

# (H,W) -> (B=1, C=1, H, W), [0,1]
img_tensor = torch.from_numpy(img).float().div(255.0).unsqueeze(0).unsqueeze(0)

with torch.no_grad():
    enc_feat = encoder(img_tensor)   # (B, D, H', W')
    print("Original image shape:", img.shape)
    print("Input tensor shape  :", img_tensor.shape)
    print("Encoder feature map :", enc_feat.shape)

# 2) 2D -> 시퀀스(열 기준): (B, D, H', W') -> (B, N, D)
if enc_feat.dim() == 4:
    B, D, Hp, Wp = enc_feat.shape
    enc_seq = enc_feat.permute(0, 3, 2, 1).contiguous().view(B, Wp * Hp, D)  # (B, N, D)
elif enc_feat.dim() == 3:
    enc_seq = enc_feat
    B, N, D = enc_seq.shape
else:
    raise RuntimeError(f"Unexpected encoder output shape: {enc_feat.shape}")

B, N, D = enc_seq.shape
print("Encoder seq shape: ", enc_seq.shape)

# 3) Decoder
V = 30522
L = 20
BOS_ID = 1
max_enc_T = max(8192, N + 8)

decoder = PALDecoder(vocab_size=V, d_model=D, max_enc_T=max_enc_T).eval()
tgt_input_ids = torch.full((B, L), BOS_ID, dtype=torch.long)

with torch.no_grad():
    logits = decoder(enc_seq, tgt_input_ids)  # (B, L, V)
    print("Decoder output shape:", logits.shape)
    assert logits.shape == (B, L, V)
    assert torch.isfinite(logits).all(), "Got NaN/Inf in logits"