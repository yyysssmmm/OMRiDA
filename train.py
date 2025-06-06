import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from models.dla import DLAModel
from loss.losses import DualLoss
from data.dataset import FormulaDataset
from data.utils.vocab import Vocab
from utils.seed import set_seed
from utils.metrics import token_accuracy
from utils.io_utils import save_log
from utils.plot_utils import plot_loss_curve

# ✅ 하이퍼파라미터
BATCH_SIZE = 8
EPOCHS = 100
LEARNING_RATE = 1.0
MATCH_WEIGHT = 1.0
IGNORE_IDX = 0
PATIENCE_THRESHOLD = 15
SAVE_DIR = "checkpoints"
BATCH_LOG_DIR = "batch_logs"

# ✅ device (1줄)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"✅ Using device: {DEVICE}")

# ✅ 시드 고정
set_seed(42)

def get_formula_transform(image_type: str):
    if image_type == "crohme_hme":
        return transforms.Compose([
            transforms.Resize((128, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.077], std=[0.249])
        ])
    elif image_type == "crohme_pme":
        return transforms.Compose([
            transforms.Resize((64, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.11], std=[0.235])
        ])
    elif image_type == "unpaired_pme":
        return transforms.Compose([
            transforms.CenterCrop((2339, 1654)),
            transforms.Resize((128, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[3.7e-5], std=[0.0014])
        ])
    else:
        raise ValueError(f"Unsupported image_type: {image_type}")

# ✅ collate 함수
def formula_collate_fn(batch):
    images = torch.stack([item["image"] for item in batch], dim=0)
    formulas = torch.nn.utils.rnn.pad_sequence(
        [item["formula"] for item in batch],
        batch_first=True,
        padding_value=IGNORE_IDX
    )
    return {"image": images, "formula": formulas}

# ✅ vocab
vocab_path = Path("data/vocab/im2latex_vocab.txt")
vocab = Vocab.load_from_txt(vocab_path)
vocab_size = len(vocab)

# ✅ Paired Dataset
hme_paired_transform = get_formula_transform("crohme_hme")
pme_paired_transform = get_formula_transform("crohme_pme")
crohme_path = Path("data/CROHME/data_crohme/train")
crohme_caption_path = crohme_path / "caption.txt"
paired_hme_dataset = FormulaDataset(str(crohme_path / "img"), str(crohme_caption_path), transform=hme_paired_transform, image_ext="bmp", vocab=vocab)
paired_pme_dataset = FormulaDataset(str(crohme_path / "pme_img"), str(crohme_caption_path), transform=pme_paired_transform, image_ext="png", vocab=vocab)

# ✅ Unpaired PME Dataset
pme_unpaired_transform = get_formula_transform("unpaired_pme")
unpaired_caption_path = "data/IM2LATEX/caption/unpaired_caption.txt"
unpaired_pme_dataset = FormulaDataset("data/IM2LATEX/img/pme_unpaired", unpaired_caption_path, transform=pme_unpaired_transform, image_ext="png", vocab=vocab)

# ✅ DataLoader
paired_loader = zip(
    DataLoader(paired_pme_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=formula_collate_fn),
    DataLoader(paired_hme_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=formula_collate_fn)
)
unpaired_loader = DataLoader(unpaired_pme_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=formula_collate_fn)

# ✅ 모델, 손실함수, 옵티마이저
model = DLAModel(vocab_size=vocab_size).to(DEVICE)
criterion = DualLoss(match_weight=MATCH_WEIGHT, ignore_index=IGNORE_IDX)
optimizer = torch.optim.Adadelta(model.parameters(), lr=LEARNING_RATE)

# ✅ 학습 루프
best_exprate = 0.0
patience = 0
loss_history = []
log_dict = {"train": []}

os.makedirs(BATCH_LOG_DIR, exist_ok=True)

total_batches = min(len(paired_pme_dataset), len(paired_hme_dataset), len(unpaired_pme_dataset)) // BATCH_SIZE

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0

    loop = tqdm(zip(paired_loader, unpaired_loader), total=total_batches, leave=True, desc=f"Epoch {epoch}/{EPOCHS}")

    for (batch_pme, batch_hme), batch_up in loop:
        img_pme = batch_pme["image"].to(DEVICE)
        tgt_pme = batch_pme["formula"].to(DEVICE)

        img_hme = batch_hme["image"].to(DEVICE)
        tgt_hme = batch_hme["formula"].to(DEVICE)

        img_up = batch_up["image"].to(DEVICE)
        tgt_up = batch_up["formula"].to(DEVICE)

        logits_p, logits_h, context_p, context_h = model(
            images_pme=img_pme, images_hme=img_hme,
            tgt_seq_pme=tgt_pme, tgt_seq_hme=tgt_hme
        )

        logits_up, _, _, _ = model(
            images_pme=img_up, images_hme=None,
            tgt_seq_pme=tgt_up, tgt_seq_hme=None,
            pme_only=True
        )

        loss, loss_dict = criterion(
            logits_h=logits_h, targets_h=tgt_hme,
            logits_p=logits_p, targets_p=tgt_pme,
            logits_up=logits_up, targets_up=tgt_up,
            context_h=context_h, context_p=context_p
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        preds = logits_h.argmax(dim=-1)
        acc = token_accuracy(preds, tgt_hme[:, 1:])

        total_loss += loss.item()
        total_acc += acc
        num_batches += 1

        batch_save_path = os.path.join(BATCH_LOG_DIR, f"epoch{epoch:03d}_batch{num_batches:03d}.pt")
        torch.save({
            "epoch": epoch,
            "batch_idx": num_batches,
            "loss_dict": loss_dict,
            "loss": loss.item(),
            "token_accuracy": acc
        }, batch_save_path)

        loop.set_postfix(loss=loss.item(), acc=acc)

    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches
    print(f"\n[Epoch {epoch}] Loss: {avg_loss:.4f}, Token Accuracy: {avg_acc:.4f}")

    log_dict["train"].append({
        "epoch": epoch,
        "loss": avg_loss,
        "token_accuracy": avg_acc
    })
    loss_history.append(avg_loss)

    current_exprate = avg_acc
    if current_exprate > best_exprate:
        best_exprate = current_exprate
        patience = 0
        os.makedirs(SAVE_DIR, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pth"))
        print("🧠 Best model saved.")
    else:
        patience += 1
        if patience >= PATIENCE_THRESHOLD:
            print(f"🛑 Early stopping at epoch {epoch}")
            break

# ✅ 최종 저장
os.makedirs(SAVE_DIR, exist_ok=True)
save_log(log_dict, save_path=os.path.join(SAVE_DIR, "train_log.json"))
plot_loss_curve(loss_history, save_path=os.path.join(SAVE_DIR, "loss_curve.png"))
torch.save(model.state_dict(), os.path.join(SAVE_DIR, "final_model.pth"))
print("✅ 최종 모델 저장 완료!")
