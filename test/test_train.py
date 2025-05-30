import sys
from pathlib import Path
root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))  # 상위 폴더 OMRiDA를 경로에 추가

import torch
from models.dla import DLAModel
from data.dataset import FormulaDataset
from data.vocab import Vocab
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images)
    labels = pad_sequence(labels, batch_first=True, padding_value=0)
    return images, labels


def test_train_step():
    # ✅ 경로 설정
    image_dir = root / "data/CROHME/data_crohme/train/img"
    caption_path = root / "data/CROHME/data_crohme/train/caption.txt"

    # ✅ vocab 구축
    vocab = Vocab()
    with open(caption_path, "r", encoding="utf-8") as f:
        formulas = [line.strip().split('\t')[1] for line in f if '\t' in line]
    vocab.build_vocab(formulas)

    # ✅ transform 정의
    transform = transforms.Compose([
        transforms.Resize((128, 512)),
        transforms.ToTensor()
    ])

    dataset = FormulaDataset(image_dir=image_dir, caption_path=caption_path, vocab=vocab, transform=transform)
    loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    # ✅ 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    model = DLAModel(vocab_size=len(vocab)).to(device)
    model.train()

    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)

    print("📷 Image shape:", images.shape)
    print("📝 Label shape:", labels.shape)

    outputs = model(images, labels)  # (B, T-1, vocab_size)
    print("🔮 Output shape:", outputs.shape)

    logits = outputs.reshape(-1, outputs.size(-1))                      # (B×(T-1), vocab)
    targets = labels[:, 1:outputs.size(1)+1].reshape(-1)                # (B×(T-1))
    print("✅ Sample loss:", torch.nn.functional.cross_entropy(logits, targets, ignore_index=0))


if __name__ == "__main__":
    test_train_step()
