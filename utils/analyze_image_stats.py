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
    names = []

    for i, img_path in enumerate(sorted(image_dir.glob(f"*.{image_ext}"))):
        if i >= sample_limit:
            break
        try:
            with Image.open(img_path) as img:
                sizes.append(img.size)
                names.append(img_path.name)
                img_tensor = transforms.ToTensor()(img)  # (C, H, W)
                means.append(torch.mean(img_tensor, dim=(1,2))) # (3, )
                stds.append(torch.std(img_tensor, dim=(1,2)))   # (3, )
        except Exception as e:
            print(f"❌ {img_path.name}: {e}")

    means = torch.stack(means)
    stds = torch.stack(stds)
    channel_mean = torch.mean(means, dim=0) # (3, )
    channel_std = torch.mean(stds, dim=0)   # (3, )

    widths, heights = zip(*sizes)
    print(len(sizes))
    print(f"Max size: {(np.max(widths), np.max(heights))}")
    print(f"Min size: {(np.min(widths), np.min(heights))}")
    print(f"Mean size: {(np.mean(widths), np.mean(heights))}")
    print(f"Median size: {(np.median(widths), np.median(heights))}")
    print(f"Top size: {Counter(sizes).most_common(5)}")
    print(f"Channel-wise Mean (R, G, B): {channel_mean.tolist()}")
    print(f"Channel-wise Std  (R, G, B): {channel_std.tolist()}")
    print() 
    return widths, heights, channel_mean, channel_std, names

if __name__ == "__main__":
    datasets = [
        {
            "name": "CROHME HME",
            "path": "../data/preprocessed/train/paired/crohme/hme_preprocessed",
            "image_ext":"png"
        },
        {
            "name": "CROHME PME",
            "path": "../data/preprocessed/train/paired/crohme/pme_preprocessed",
            "image_ext":"png"
        },
        {
            "name": "IM2LATEX paired HME",
            "path": "../data/preprocessed/train/paired/im2latex/hme",
            "image_ext":"png"
        },
        {
            "name": "IM2LATEX paired PME",
            "path": "../data/preprocessed/train/paired/im2latex/pme_cropped",
            "image_ext":"png"
        }
    ]

    for ds in datasets:
        print(f"📂 분석 중: {ds['name']}")
        widths, heights, means, stds, names = analyze_image_stats(
            image_dir=ds["path"], image_ext=ds["image_ext"]
        )
        print(names[np.argmax(widths)])
        print(names[np.argmax(heights)])