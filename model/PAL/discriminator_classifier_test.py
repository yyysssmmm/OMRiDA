import os
import cv2
import numpy as np
import torch
from multi_directional_encoder import PALEncoder
from discriminator import Discriminator
from classifier import Classifier

# 한글 경로 읽기
def imread_unicode(path, flags=cv2.IMREAD_GRAYSCALE):
    stream = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(stream, flags)
    return img

# Discriminator 테스트 (아직 학습 안된 상태라 구조, 입출력 처리만 확인)
encoder = PALEncoder()
discriminator = Discriminator(input_dim=256)
encoder.eval()
discriminator.eval()

# 이미지 경로 지정
base_dir = "C:/Users/sophi/OneDrive/바탕 화면/Folder/2025 HYU/인공지능프로젝트/data/crohme/dummy"
crohme_path = os.path.normpath(os.path.join(base_dir,"hme_preprocessed/90_carlos.png"))  # 예시 파일명
cropme_path = os.path.normpath(os.path.join(base_dir,"pme_preprocessed/90_carlos.png"))  # 예시 파일명

# 입력 이미지 전처리
def preprocess_image(path):
    img = imread_unicode(path)
    if img is None:
        raise FileNotFoundError(f"Missing image: {path}")
    img_tensor = torch.from_numpy(img).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    return img_tensor

img1 = preprocess_image(crohme_path)   # handwritten
img2 = preprocess_image(cropme_path)   # printed

# Feature 추출
with torch.no_grad():
    feat1 = encoder(img1)  # (1, T, D)
    feat2 = encoder(img2)

    pred1 = discriminator(feat1)  # (1, T)
    pred2 = discriminator(feat2)

    print(f"CROHME domain prediction (shape): {pred1.shape}")
    print(f"CROPME domain prediction (shape): {pred2.shape}")
    print(f"HME pred: {pred1[0, :5]}")
    print(f"PME pred: {pred2[0, :5]}")

# Classifier 테스트
classifier = Classifier(d_model=256, vocab_size=30522)
classifier.eval()

with torch.no_grad():
    class_logits_1 = classifier(feat1)  # feat1: (1, 68, 256)
    print("Classifier output shape for hme:", class_logits_1.shape)
    class_logits_2 = classifier(feat2)  # feat2: (1, 68, 256)
    print("Classifier output shape for pme:", class_logits_2.shape)