import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import yaml
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from utils.data_utils import get_formula_transform

# ✅ config.yaml 불러오기
def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="../config.yaml")
args = parser.parse_args()
config = load_config(args.config)


def visualize_transform(image_path, image_type):
    transform = get_formula_transform(image_type, config["transforms"])

    img = Image.open(image_path)
    # ⛔ 경고 방지용 모드 정규화
    if img.mode == "P":
        img = img.convert("RGBA")
    img = img.convert("RGB")  # PME, HME 모두 RGB로 처리

    print(img.size)
    # transform 중 Normalize 제외하고 시각화용으로 다시 구성
    transform_for_display = transforms.Compose([
        t for t in transform.transforms if not isinstance(t, transforms.Normalize)
    ])

    transformed_img = transform_for_display(img)  # Tensor (C, H, W)

    # Tensor → numpy for plotting
    img_np = transformed_img.numpy().squeeze()
    print(img_np.shape)
    if img_np.ndim == 2:
        plt.imshow(img_np, cmap="gray")
    else:
        plt.imshow(img_np.transpose(1, 2, 0))

    plt.title(f"{image_type} transformed")
    plt.axis("off")
    plt.show()

# ✅ 테스트 실행
if __name__ == "__main__":
    #image_path = "../data/paired_CROHME+IM2LATEX/hme/crohme_00000.png"
    #image_path = "../data/paired_CROHME+IM2LATEX/hme/im2latex_68343.png"
    #visualize_transform(image_path, "paired_hme")

    #image_path = "../data/paired_CROHME+IM2LATEX/pme/im2latex_96490.png"
    #image_path = "../data/paired_CROHME+IM2LATEX/pme/crohme_02340.png"
    
    #visualize_transform(image_path, "paired_pme")

    #image_path = "../data/IM2LATEX/img/pme_unpaired/51.png"
    #image_path = "../data/CROHME/data_crohme/train/img/70_carlos.bmp"
    image_path = "../data/CROHME/data_crohme/train/pme_img/TrainData2_24_sub_43.png"
    visualize_transform(image_path, "unpaired")
