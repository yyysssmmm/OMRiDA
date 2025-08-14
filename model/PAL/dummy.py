import os
import random
import cv2
import numpy as np
import shutil

# # 랜덤으로 뽑아서 dummy 데이터셋 (caption+hme 폴더에 이미지+pme 폴더에 이미지 만들기)
# base_dir = "C:/Users/sophi/OneDrive/바탕 화면/Folder/2025 HYU/인공지능프로젝트/data/crohme"
# caption_path = os.path.join(base_dir, "caption.txt")
# with open(caption_path, "r", encoding="utf-8") as f:
#     captions = f.readlines()

# # 각 구간에서 랜덤하게 추출 (앞에서 뒤로 갈수록 복잡한 수식)
# sampled = (
#     random.sample(captions[0:1000], 10) +
#     random.sample(captions[3000:4000], 10) +
#     random.sample(captions[7000:], 10)
# )

# # caption_dummy.txt 저장
# output_path = os.path.join(base_dir, "dummy/caption.txt")
# with open(output_path, "w", encoding="utf-8") as f:
#     f.writelines(sampled)

# 경로 설정
base_dir = "C:/Users/sophi/OneDrive/바탕 화면/Folder/2025 HYU/인공지능프로젝트/data/crohme"
caption_path = os.path.join(base_dir, "dummy/caption.txt")
crohme = os.path.join(base_dir, "hme_preprocessed")
cropme = os.path.join(base_dir, "pme_preprocessed")
crohme_dummy = os.path.join(base_dir, "dummy/hme_preprocessed")
cropme_dummy = os.path.join(base_dir, "dummy/pme_preprocessed")
os.makedirs(crohme_dummy, exist_ok=True)
os.makedirs(cropme_dummy, exist_ok=True)

with open(caption_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

for line in lines:
    img_name = line.strip().split()[0] + ".png"
    
    crohme_path = os.path.normpath(os.path.join(crohme, img_name))
    cropme_path = os.path.normpath(os.path.join(cropme, img_name))
    
    crohme_dummy_path = os.path.normpath(os.path.join(crohme_dummy, img_name))
    cropme_dummy_path = os.path.normpath(os.path.join(cropme_dummy, img_name))
    
    if os.path.exists(crohme_path):
        shutil.copy2(crohme_path, crohme_dummy_path)
    else:
        print(f"{crohme_path} not found.")

    if os.path.exists(cropme_path):
        shutil.copy2(cropme_path, cropme_dummy_path)
    else:
        print(f"{cropme_path} not found.")