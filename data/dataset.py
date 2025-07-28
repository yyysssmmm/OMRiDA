from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
import torch


class FormulaDataset(Dataset):
    def __init__(self, image_dir, caption_path, transform=None, image_ext="png", vocab=None):
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

                if '.' not in file_id:
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
        image = Image.open(image_path).convert('L')

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

class PairedFormulaDataset(Dataset):
    def __init__(self, hme_dir, pme_dir, caption_path, transform_hme, transform_pme, image_exts=["png", "png"], vocab=None):
        self.hme_dir = Path(hme_dir)
        self.pme_dir = Path(pme_dir)
        self.transform_hme = transform_hme
        self.transform_pme = transform_pme
        self.vocab = vocab
        self.img_exts = image_exts

        self.samples = []
        with open(caption_path, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) != 2:
                    continue
                img_name, formula = parts 
                self.samples.append((img_name, formula))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, formula = self.samples[idx]

        if '.' not in img_name:
            img_hme_path = self.hme_dir / f"{img_name}.{self.img_exts[0]}"
            img_pme_path = self.pme_dir / f"{img_name}.{self.img_exts[1]}"

        else:
            img_hme_path = self.hme_dir / img_name
            img_pme_path = self.pme_dir / img_name

        img_hme = Image.open(img_hme_path).convert('L')
        img_pme = Image.open(img_pme_path).convert('L')

        if self.transform_hme:
            img_hme = self.transform_hme(img_hme)
        if self.transform_pme:
            img_pme = self.transform_pme(img_pme)

        token_ids = self.vocab.encode(formula)
        formula_tensor = torch.tensor(token_ids, dtype=torch.long)

        return {
            "img_hme": img_hme,
            "img_pme": img_pme,
            "formula": formula_tensor
        }
    
    

