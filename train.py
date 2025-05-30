import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path

from models.dla import DLAModel
from data.dataset import FormulaDataset
from data.vocab import Vocab


def collate_fn(batch):
    """
    커스텀 collate function.
    - 이미지: (B, 3, H, W)로 스택
    - 라벨: padding해서 (B, T_max) 텐서로 만듦
    """
    from torch.nn.utils.rnn import pad_sequence

    images, labels = zip(*batch)
    images = torch.stack(images)
    labels = pad_sequence(labels, batch_first=True, padding_value=0)

    return images, labels


# ✅ 하이퍼파라미터
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-3
MAX_SEQ_LEN = 100

# ✅ 경로 설정
root = Path(__file__).resolve().parent
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

# ✅ 데이터셋 & 로더
train_dataset = FormulaDataset(
    image_dir=image_dir,
    caption_path=caption_path,
    vocab=vocab,
    transform=transform
)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    collate_fn=collate_fn
)

# ✅ 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# ✅ 모델
model = DLAModel(vocab_size=len(vocab)).to(device)

# ✅ Loss 및 Optimizer
criterion = nn.CrossEntropyLoss(ignore_index=0)  # <pad> 무시
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ✅ 학습 루프
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for images, token_ids in train_loader:
        images = images.to(device)               # (B, 3, H, W)
        token_ids = token_ids.to(device)         # (B, T)

        optimizer.zero_grad()

        # decoder는 <sos> ~ <eos> 전까지만 입력받고, 그 다음 토큰을 예측함
        outputs = model(images, token_ids)       # outputs.shape: (B, T-1, vocab_size)

        # 정답은 <sos> 다음부터
        logits = outputs.reshape(-1, outputs.size(-1))           # (B×(T-1), vocab_size)
        targets = token_ids[:, 1:outputs.size(1)+1].reshape(-1)  # (B×(T-1))

        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"[Epoch {epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f}")

# ✅ 모델 저장
torch.save(model.state_dict(), "dla_model.pt")
print("✅ 모델 저장 완료: dla_model.pt")
