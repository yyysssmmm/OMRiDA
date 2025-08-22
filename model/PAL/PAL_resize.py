import os
import cv2
import numpy as np
from tqdm import tqdm

# 한글 경로 읽기/저장
def imread_unicode(path, flags=cv2.IMREAD_GRAYSCALE):
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)

def imwrite_unicode(path, img, ext=".png", params=None):
    if params is None:
        params = []
    success, buf = cv2.imencode(ext, img, params)
    if not success:
        return False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    buf.tofile(path)
    return True

# 경로 설정
base_dir = "C:/Users/sophi/OneDrive/바탕 화면/Folder/2025 HYU/인공지능프로젝트/data/crohme"
input_dir = "C:/Users/sophi/OneDrive/바탕 화면/Folder/2025 HYU/인공지능프로젝트/data/crohme/hme_preprocessed"
output_dir = "C:/Users/sophi/Documents/hme_preprocessed_resized"
os.makedirs(output_dir, exist_ok=True)

target_height = 128

def resize_keep_aspect(img, target_height):
    h, w = img.shape[:2]
    if h == 0:
        return None
    scale = target_height / float(h)
    new_w = int(round(w * scale))
    new_w = max(1, new_w)
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(img, (new_w, target_height), interpolation=interp)

# caption.txt에 나온 이미지들만 대상으로 전처리
caption_path = os.path.join(base_dir, "caption.txt")
with open(caption_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

image_names = [line.strip().split()[0] + ".png" for line in lines]

for img_name in tqdm(image_names, desc="Resizing images"):
    input_path = os.path.normpath(os.path.join(input_dir, img_name))
    output_path = os.path.normpath(os.path.join(output_dir, img_name))

    if not os.path.exists(input_path):
        print(f"Missing file: {input_path}")
        continue

    img = imread_unicode(input_path, flags=cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Failed to load: {input_path}")
        continue

    # 혹시 흰 배경/검정 글자 보장하고 싶다면 여기서 이진화/반전 로직 옵션으로 추가 가능
    # _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    resized = resize_keep_aspect(img, target_height)
    if resized is None:
        print(f"Bad shape: {input_path}")
        continue

    if not imwrite_unicode(output_path, resized, ext=".png"):
        print(f"Failed to write: {output_path}")