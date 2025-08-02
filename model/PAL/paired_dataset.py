import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class PairedDataset(Dataset):
    def __init__(self, list_path, img_dir_h, img_dir_p, transform=None):
        """
        list_path: str, txt/csv 파일. 각 줄에 이미지 파일명과 label이 있는 구조
        img_dir_h: str, handwritten 이미지 경로
        img_dir_p: str, printed 이미지 경로
        """
        with open(list_path, 'r') as f:
            self.data = [line.strip().split('\t') for line in f] 
        self.img_dir_h = img_dir_h
        self.img_dir_p = img_dir_p
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_h_path, img_p_path, label = self.data[idx]
        img_h = Image.open(os.path.join(self.img_dir_h, img_h_path)).convert("L")
        img_p = Image.open(os.path.join(self.img_dir_p, img_p_path)).convert("L")
        if self.transform:
            img_h = self.transform(img_h)
            img_p = self.transform(img_p)

        label = [int(token) for token in label.split()]  # 예: "12 25 13 4" → [12, 25, 13, 4]
        label = torch.tensor(label, dtype=torch.long)

        return img_p, img_h, label, label  # x_p, x_h, y_p, y_h