import os
import cv2
import numpy as np
import torch
from multi_directional_encoder import PALEncoder
from conv_attention_decoder import PALDecoder

# 한글 경로 읽기
def imread_unicode(path, flags=cv2.IMREAD_GRAYSCALE):
    stream = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(stream, flags)
    return img

# Encoder 테스트
encoder = PALEncoder()
encoder.eval()

base_dir = "C:/Users/sophi/OneDrive/바탕 화면/Folder/2025 HYU/인공지능프로젝트/data/crohme/dummy"
img_path = os.path.join(base_dir, "hme_preprocessed")

# 이미지 로딩
image_path = os.path.normpath(os.path.join(img_path,"90_carlos.png"))  # 예시 파일명
img = imread_unicode(image_path)

if img is None:
    print("Image not found!")
else:
    # 전처리: 정규화 및 차원 확장
    img_tensor = torch.from_numpy(img).float() / 255.0  # [0, 1] 정규화
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)   # (1, 1, H, W)

    with torch.no_grad():
        out = encoder(img_tensor)  # (1, T, D)
        print("Original image shape: ",img.shape)
        print("Input tensor shape: ", img_tensor.shape)
        print("Encoder output shape:", out.shape)

# Decoder 테스트
decoder = PALDecoder(d_model=256, vocab_size=30522)
decoder.eval()

# encoder 출력 그대로 사용
with torch.no_grad():
    logits = decoder(out)
    print("Decoder output shape:", logits.shape)
