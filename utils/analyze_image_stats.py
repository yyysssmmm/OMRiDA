import os
from pathlib import Path
from PIL import Image
from collections import Counter
import numpy as np
import torch
from torchvision import transforms

def analyze_image_stats(image_dir, image_ext="png", sample_limit=1000000):
    image_dir = Path(image_dir)
    sizes = []
    means = []
    stds = []

    for i, img_path in enumerate(sorted(image_dir.glob(f"*.{image_ext}"))):
        if i >= sample_limit:
            break
        try:
            with Image.open(img_path) as img:
                sizes.append(img.size)

                # 그레이스케일로 변환 후 numpy 배열로 전환
                img_tensor = transforms.ToTensor()(img.convert("L"))  # (1, H, W)
                means.append(torch.mean(img_tensor).item())
                stds.append(torch.std(img_tensor).item())
        except Exception as e:
            print(f"❌ {img_path.name}: {e}")

    widths, heights = zip(*sizes)
    print(len(sizes))
    print(f"Max size: {(np.max(widths), np.max(heights))}")
    print(f"min size: {(np.min(widths), np.min(heights))}")
    print(f"Mean size: {(np.mean(widths), np.mean(heights))}")
    print(f"Median size: {(np.median(widths), np.median(heights))}")
    print(f"Top size: {Counter(sizes).most_common(5)}")
    print(f"Pixel Mean (grayscale): {np.mean(means):.4f}")
    print(f"Pixel Std  (grayscale): {np.mean(stds):.4f}")
    print() 

if __name__ == "__main__":
    datasets = [
        {
            "name": "CROHME HME",
            "path": "../data/preprocessed/train/paired/crohme/hme",
            "image_ext":"bmp"
        },
        {
            "name": "CROHME PME",
            "path": "../data/preprocessed/train/paired/crohme/pme",
            "image_ext":"png"
        },
        {
            "name": "IM2LATEX paired HME",
            "path": "../data/preprocessed/train/paired/im2latex/hme",
            "image_ext":"png"
        },
        {
            "name": "IM2LATEX paired PME",
            "path": "../data/preprocessed/train/paired/im2latex/pme",
            "image_ext":"png"
        }
    ]

    for ds in datasets:
        print(f"📂 분석 중: {ds['name']}")
        analyze_image_stats(
            image_dir=ds["path"], image_ext=ds["image_ext"]
        )
