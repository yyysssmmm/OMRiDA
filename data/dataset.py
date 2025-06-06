from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
import torch


class FormulaDataset(Dataset):
    def __init__(self, image_dir, caption_path, transform=None, image_ext="bmp", vocab=None, add_ext_if_missing=True):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.image_ext = image_ext
        self.vocab = vocab
        self.add_ext_if_missing = add_ext_if_missing

        self.samples = []
        with open(caption_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) != 2:
                    continue
                file_id, latex = parts

                # 확장자 자동 부여 여부 결정
                if self.add_ext_if_missing and '.' not in file_id:
                    filename = f"{file_id}.{self.image_ext}"
                else:
                    filename = file_id

                image_path = self.image_dir / filename
                if image_path.exists():
                    self.samples.append((image_path, latex))

    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        image_path, latex = self.samples[idx]
        image = Image.open(image_path)

        # 🛠 경고 제거용 추가 처리
        if image.mode == "P":
            image = image.convert("RGBA")
        image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        formula_ids = self.vocab.encode(latex)
        
        # ⚠️ 최소 길이 2 이상만 허용 (sos + token 또는 token + eos)
        if len(formula_ids) < 2:
            # 다음 인덱스로 순환 접근
            return self.__getitem__((idx + 1) % len(self.samples))

        formula_tensor = torch.tensor(formula_ids)
        return {
            "image": image,
            "formula": formula_tensor
        }
