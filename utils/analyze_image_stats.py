# analyze_image_stats.py

import os
from pathlib import Path
from PIL import Image
from collections import Counter
import numpy as np
import torch
from torchvision import transforms 

def analyze_image_sizes(image_dir, image_ext="png", sample_limit=100):
    image_dir = Path(image_dir)
    sizes = []

    files = sorted(image_dir.glob(f"*.{image_ext}"))
    for i, img_path in enumerate(files):
        if i >= sample_limit:
            break
        with Image.open(img_path) as img:
            sizes.append(img.size)  # (width, height)

    widths, heights = zip(*sizes)
    print(f"✅ Sampled {len(sizes)} images from {image_dir}")
    print(f"📐 Width  - min: {min(widths)}, max: {max(widths)}, mean: {np.mean(widths):.1f}, median: {np.median(widths)}")
    print(f"📏 Height - min: {min(heights)}, max: {max(heights)}, mean: {np.mean(heights):.1f}, median: {np.median(heights)}")

    # Top-5 frequent sizes
    counter = Counter(sizes)
    print(f"\n🎯 Top 5 most common sizes:")
    for size, count in counter.most_common(5):
        print(f"  {size}: {count} images")

def compute_image_mean_std(image_dir, image_ext="png", sample_limit=100, resize=(128, 512)):
    image_dir = Path(image_dir)
    images = []

    files = sorted(image_dir.glob(f"*.{image_ext}"))
    tf = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor()
    ])

    for i, img_path in enumerate(files):
        if i >= sample_limit:
            break
        with Image.open(img_path) as img:
            img_tensor = tf(img)  # 자동으로 (C,H,W)
            if img_tensor.ndim == 2:
                img_tensor = img_tensor.unsqueeze(0)
            images.append(img_tensor)

    images_tensor = torch.stack(images, dim=0)  # (N, C, H, W)
    mean = images_tensor.mean(dim=[0, 2, 3])
    std = images_tensor.std(dim=[0, 2, 3])

    print(f"✅ Mean: {mean.tolist()}")
    print(f"✅ Std : {std.tolist()}")

if __name__ == "__main__":
    print("\n--- CROHME HME Images ---")
    analyze_image_sizes("../data/CROHME/data_crohme/train/img", image_ext="bmp", sample_limit=1000000)
    compute_image_mean_std("../data/CROHME/data_crohme/train/img", image_ext="bmp", sample_limit=1000000)

    print("\n--- CROHME PME Images ---")
    analyze_image_sizes("../data/CROHME/data_crohme/train/pme_img", image_ext="png", sample_limit=1000000)
    compute_image_mean_std("../data/CROHME/data_crohme/train/pme_img", image_ext="png", sample_limit=1000000)

    print("\n--- IM2LATEX Unpaired PME ---")
    analyze_image_sizes("../data/IM2LATEX/img/pme_unpaired", image_ext="png", sample_limit=1000000)
    compute_image_mean_std("../data/IM2LATEX/img/pme_unpaired", image_ext="png", sample_limit=1000000)
