import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image

# vocab 불러오기
def load_vocab(vocab_path):
    with open(vocab_path, encoding="utf-8") as f:
        tokens = [ln.strip() for ln in f if ln.strip()]
    token2id = {tok: i for i, tok in enumerate(tokens)}
    pad_id = token2id["<pad>"]
    sos_id = token2id.get("<sos>") or token2id.get("<bos>")
    eos_id = token2id["<eos>"]
    unk_id = token2id.get("<unk>")
    return token2id, pad_id, sos_id, eos_id, unk_id

# label_to_ids 만들기
def make_label_to_ids(token2id, sos_id=None, eos_id=None, unk_id=None,
                      add_sos=True, add_eos=True):
    def _fn(label_str: str):
        toks = label_str.strip().split()
        ids = [token2id.get(t, unk_id) if unk_id is not None else token2id[t] for t in toks]
        if add_sos and sos_id is not None:
            ids = [sos_id] + ids
        if add_eos and eos_id is not None:
            ids = ids + [eos_id]
        return ids
    return _fn

class PairedDataset(Dataset):
    def __init__(self, list_path, img_dir_h, img_dir_p, transform=None, label_to_ids=None):
        """
        list_path: str, txt/csv 파일. 각 줄에 이미지 파일명과 label이 있는 구조
        img_dir_h: str, handwritten 이미지 경로
        img_dir_p: str, printed 이미지 경로
        transform: ToTensor()+Normalize 등
        label_to_ids: str -> List[int]
        """
        self.img_dir_h = img_dir_h
        self.img_dir_p = img_dir_p
        self.transform = transform
        self.label_to_ids = label_to_ids

        self.data = []
        with open(list_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) == 1:
                    continue
                img_id, label_str = parts[0], parts[1]
                self.data.append((img_id, label_str))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_id, label_str = self.data[idx]

        h_path = os.path.normpath(os.path.join(self.img_dir_h, img_id + ".png"))
        p_path = os.path.normpath(os.path.join(self.img_dir_p, img_id + ".png"))

        img_h = Image.open(h_path).convert("L")
        img_p = Image.open(p_path).convert("L")

        if self.transform:
            img_h = self.transform(img_h)
            img_p = self.transform(img_p)

        if self.label_to_ids is not None:
            ids = self.label_to_ids(label_str)  # -> List[int]
        else:
            try:
                ids = [int(tok) for tok in label_str.split()]
            except ValueError:
                raise ValueError("label_to_ids Callback")
        label = torch.tensor(ids, dtype=torch.long)

        return img_p, img_h, label, label  # x_p, x_h, y_p, y_h
    
# # 작동 테스트 부분 (추후 삭제)
# base_dir = "C:/Users/sophi/OneDrive/바탕 화면/Folder/2025 HYU/인공지능프로젝트/data/crohme/dummy"
# list_path = os.path.normpath(os.path.join(base_dir,"caption.txt"))
# img_dir_h = os.path.normpath(os.path.join(base_dir,"hme_preprocessed"))
# img_dir_p = os.path.normpath(os.path.join(base_dir,"pme_preprocessed"))
# base_dir = "C:/Users/sophi/OneDrive/바탕 화면/Folder/2025 HYU/인공지능프로젝트/data/crohme"
# vocab_path = os.path.normpath(os.path.join(base_dir, "crohme_vocab.txt"))
# token2id, pad_id, sos_id, eos_id, unk_id = load_vocab(vocab_path)
# label_to_ids = make_label_to_ids(token2id, sos_id, eos_id, unk_id, add_sos=True, add_eos=True)
# transform = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])

# dataset = PairedDataset(list_path, img_dir_h, img_dir_p, transform=transform, label_to_ids=label_to_ids)
# print(f"Loaded dataset with {len(dataset)} samples.")

# # 첫 샘플 확인
# x_p, x_h, y_p, y_h = dataset[0]
# print("Printed image tensor shape:", x_p.shape)   # (1, 128, 512)
# print("Handwritten image tensor shape:", x_h.shape)
# print("Label tensor (PME):", y_p)
# print("Label tensor (HME):", y_h)
# print("ID sequence (PME): ", y_p[:10])
# print("ID sequence (HME): ", y_h[:10])