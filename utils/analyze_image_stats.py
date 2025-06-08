import os
from pathlib import Path
from PIL import Image
from collections import Counter
import numpy as np
import torch
from torchvision import transforms

def analyze_image_stats(image_dir, image_ext="png", sample_limit=100):
    image_dir = Path(image_dir)
    sizes = []

    for i, img_path in enumerate(sorted(image_dir.glob(f"*.{image_ext}"))):
        if i >= sample_limit:
            break
        try:
            with Image.open(img_path) as img:
                sizes.append(img.size)
        except Exception as e:
            print(f"❌ {img_path.name}: {e}")

    widths, heights = zip(*sizes)
    print(f"Max size: {(np.max(widths), np.max(heights))}")
    print(f"Mean size: {(np.mean(widths), np.mean(heights))}")
    print(f"Median size: {(np.median(widths), np.median(heights))}")
    print(f"Top size: {Counter(sizes).most_common(5)}")
    print() 


if __name__ == "__main__":
    datasets = [
        {
            "name": "CROHME HME",
            "path": "../data/CROHME/data_crohme/train/img",
            "image_ext":"bmp"
        },
        {
            "name": "CROHME PME",
            "path": "../data/CROHME/data_crohme/train/pme_img",
            "image_ext":"png"
        },
        {
            "name": "IM2LATEX paired PME",
            "path": "../data/IM2LATEX/img/pme_paired",
            "image_ext":"png"
        },
        {
            "name": "IM2LATEX paired HME",
            "path": "../data/IM2LATEX/img/hme_paired",
            "image_ext":"png"
        }
    ]

    for ds in datasets:
        print(ds["name"])

        analyze_image_stats(
            image_dir=ds["path"], image_ext=ds["image_ext"]
        )
        print()
