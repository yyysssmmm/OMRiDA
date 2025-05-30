import os
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
import torch


class FormulaDataset(Dataset):
    def __init__(self, image_dir, caption_path, transform=None, image_ext="bmp", vocab=None):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.image_ext = image_ext
        self.vocab = vocab

        self.samples = []
        with open(caption_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) != 2:
                    continue
                file_id, latex = parts
                image_path = self.image_dir / f"{file_id}.{self.image_ext}"
                if image_path.exists():
                    self.samples.append((image_path, latex))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, latex = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(self.vocab.encode(latex))
