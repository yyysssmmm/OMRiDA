import sys
from pathlib import Path
root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))  # 상위 폴더 OMRiDA를 경로에 추가

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from models.dla import DLAModel
from data.dataset import FormulaDataset
from data.vocab import Vocab


def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images)
    labels = pad_sequence(labels, batch_first=True, padding_value=0)
    return images, labels


def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, total_epochs, log_interval=100, grad_clip=None):
    model.train()
    total_loss = 0

    loop = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"🚀 Epoch [{epoch}/{total_epochs}]", leave=True)

    for batch_idx, (images, token_ids) in loop:
        images, token_ids = images.to(device), token_ids.to(device)
        optimizer.zero_grad()

        outputs = model(images, token_ids)  # (B, T-1, vocab_size)
        logits = outputs.reshape(-1, outputs.size(-1))           # (B×(T-1), vocab_size)
        targets = token_ids[:, 1:outputs.size(1)+1].reshape(-1)  # (B×(T-1))

        loss = criterion(logits, targets)
        loss.backward()

        if grad_clip:
            clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        total_loss += loss.item()

        loop.set_postfix(loss=loss.item())

        if (batch_idx + 1) % log_interval == 0:
            print(f"[Epoch {epoch}/{total_epochs} | Batch {batch_idx+1}/{len(dataloader)}] Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    print(f"✅ [Epoch {epoch}] Avg Loss: {avg_loss:.4f}")
    return avg_loss


def save_checkpoint(model, optimizer, epoch, path="dla_model.pt"):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }, path)
    print(f"💾 모델 저장 완료: {path}")


# ✅ 실행용 메인 블록
if __name__ == "__main__":

    # 🔧 하이퍼파라미터
    BATCH_SIZE = 16
    EPOCHS = 10
    LEARNING_RATE = 1e-3

    # 📂 경로 설정
    image_dir = "../data/CROHME/data_crohme/train/img"
    caption_path = "../data/CROHME/data_crohme/train/caption.txt"

    # 🧠 vocab 구축
    vocab = Vocab()
    with open(caption_path, "r", encoding="utf-8") as f:
        formulas = [line.strip().split('\t')[1] for line in f if '\t' in line]
    vocab.build_vocab(formulas)

    # 🖼️ transform 정의
    transform = transforms.Compose([
        transforms.Resize((128, 512)),
        transforms.ToTensor()
    ])

    # 📚 Dataset & Dataloader
    dataset = FormulaDataset(
        image_dir=image_dir,
        caption_path=caption_path,
        vocab=vocab,
        transform=transform
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, collate_fn=collate_fn)

    # 🚀 모델 및 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = DLAModel(vocab_size=len(vocab)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # 🔁 학습 루프
    for epoch in range(1, EPOCHS + 1):
        train_one_epoch(model, loader, optimizer, criterion, device, epoch, total_epochs=EPOCHS)

    # 💾 저장
    save_checkpoint(model, optimizer, epoch=EPOCHS)
