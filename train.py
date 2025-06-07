import os
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import yaml
import argparse
from datetime import datetime

from models.dla import DLAModel
from loss.losses import DualLoss
from data.dataset import FormulaDataset
from data.utils.vocab import Vocab
from utils.seed import set_seed
from utils.metrics import token_accuracy
from utils.io_utils import save_log
from utils.plot_utils import plot_loss_curve
from utils.data_utils import get_formula_transform, formula_collate_fn

# ✅ config 불러오기
def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="config.yaml")
args = parser.parse_args()
config = load_config(args.config)

# ✅ 하이퍼파라미터 기반 실험 이름 설정
EXP_NAME = (
    f"{config['experiment_name']}"
    f"_bs{config['training']['batch_size']}"
    f"_lr{config['training']['learning_rate']}"
    f"_match{config['training']['match_weight']}"
)

if config["training"]["scheduler"]["use"]:
    sched = config["training"]["scheduler"]
    EXP_NAME += f"_sched{sched['type']}{sched['step_size']}_{sched['gamma']}"

EXP_NAME += f"_seed{config['misc']['seed']}"
SAVE_ROOT = Path("runs") / EXP_NAME
SAVE_DIR = SAVE_ROOT / "checkpoints"
BATCH_LOG_DIR = SAVE_ROOT / "batch_logs"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
BATCH_LOG_DIR.mkdir(parents=True, exist_ok=True)

# ✅ 하이퍼파라미터 세팅
BATCH_SIZE = config["training"]["batch_size"]
EPOCHS = config["training"]["epochs"]
LEARNING_RATE = config["training"]["learning_rate"]
MATCH_WEIGHT = config["training"]["match_weight"]
IGNORE_IDX = config["training"]["ignore_idx"]
PATIENCE_THRESHOLD = config["training"]["early_stop_patience"]
GRAD_CLIP = config["training"]["grad_clip"]

# ✅ device 설정
DEVICE = torch.device(config["misc"]["device"] if torch.backends.mps.is_available() or torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {DEVICE}")

# ✅ 시드 고정
set_seed(config["misc"]["seed"])

# ✅ vocab
vocab = Vocab.load_from_txt(Path(config["data"]["vocab"]))
vocab_size = len(vocab)

# ✅ transform
hme_paired_transform = get_formula_transform("crohme_hme")
pme_paired_transform = get_formula_transform("crohme_pme")
pme_unpaired_transform = get_formula_transform("unpaired_pme")

# ✅ Dataset
paired_caption_path = config["data"]["paired"]["caption"]
paired_hme_dataset = FormulaDataset(config["data"]["paired"]["hme_img"], paired_caption_path, hme_paired_transform, "bmp", vocab)
paired_pme_dataset = FormulaDataset(config["data"]["paired"]["pme_img"], paired_caption_path, pme_paired_transform, "png", vocab)
unpaired_pme_dataset = FormulaDataset(config["data"]["unpaired"]["pme_img"], config["data"]["unpaired"]["caption"], pme_unpaired_transform, "png", vocab)

# ✅ Dataloader
collate_fn = lambda b: formula_collate_fn(b, pad_idx=IGNORE_IDX)
paired_hme_loader = DataLoader(paired_hme_dataset, BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
paired_pme_loader = DataLoader(paired_pme_dataset, BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
unpaired_loader = DataLoader(unpaired_pme_dataset, BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# ✅ 모델, 손실함수, 옵티마이저
model_config = config["model"]
paired_model = DLAModel(vocab_size=vocab_size, model_config=model_config, is_paired=True).to(DEVICE)
unpaired_model = DLAModel(vocab_size=vocab_size, model_config=model_config, is_paired=False).to(DEVICE)
criterion = DualLoss(match_weight=MATCH_WEIGHT, ignore_index=IGNORE_IDX)

# 🔧 optimizer 선택
optimizer_name = config["training"].get("optimizer", "adadelta").lower()
params = list(paired_model.parameters()) + list(unpaired_model.parameters())
if optimizer_name == "adam":
    optimizer = torch.optim.Adam(params, lr=LEARNING_RATE)
elif optimizer_name == "sgd":
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9)
elif optimizer_name == "adadelta":
    optimizer = torch.optim.Adadelta(params, lr=LEARNING_RATE)
else:
    raise ValueError(f"❌ Unknown optimizer: {optimizer_name}")

# ✅ learning rate scheduler
scheduler = None
sched_cfg = config["training"]["scheduler"]
if sched_cfg["use"]:
    if sched_cfg["type"] == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sched_cfg["step_size"], gamma=sched_cfg["gamma"])
    else:
        raise NotImplementedError(f"{sched_cfg['type']} scheduler is not supported.")

# ✅ 학습 루프
best_loss = float("inf")
patience = 0
loss_history, log_dict = [], {"train": []}
unpaired_iter = iter(unpaired_loader)

for epoch in range(1, EPOCHS + 1):
    paired_model.train()
    unpaired_model.train()
    total_loss, total_acc, num_batches = 0.0, 0.0, 0

    loop = tqdm(zip(paired_pme_loader, paired_hme_loader), total=len(paired_pme_loader), desc=f"Epoch {epoch}/{EPOCHS}", leave=True)
    for batch_pme, batch_hme in loop:
        try:
            batch_up = next(unpaired_iter)
        except StopIteration:
            unpaired_iter = iter(unpaired_loader)
            batch_up = next(unpaired_iter)

        img_pme, tgt_pme = batch_pme["image"].to(DEVICE), batch_pme["formula"].to(DEVICE)
        img_hme, tgt_hme = batch_hme["image"].to(DEVICE), batch_hme["formula"].to(DEVICE)
        img_up, tgt_up = batch_up["image"].to(DEVICE), batch_up["formula"].to(DEVICE)

        logits_p, logits_h, context_p, context_h = paired_model(img_pme, img_hme, tgt_pme, tgt_hme)
        logits_up, _, _, _ = unpaired_model(img_up, None, tgt_up, None)

        loss, loss_dict = criterion(logits_h, tgt_hme, logits_p, tgt_pme, logits_up, tgt_up, context_h, context_p)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
        optimizer.step()

        preds = logits_h.argmax(dim=-1)
        acc = token_accuracy(preds, tgt_hme[:, 1:])

        total_loss += loss.item()
        total_acc += acc
        num_batches += 1

        torch.save({
            "epoch": epoch,
            "batch_idx": num_batches,
            "loss_dict": loss_dict,
            "loss": loss.item(),
            "token_accuracy": acc
        }, BATCH_LOG_DIR / f"epoch{epoch:03d}_batch{num_batches:03d}.pt")

        loop.set_postfix(loss=loss.item(), acc=acc)

    if num_batches == 0:
        print("⚠️ No batch processed.")
        continue

    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches
    print(f"\n📊 [Epoch {epoch}] Loss: {avg_loss:.4f}, Token Acc: {avg_acc:.4f}")

    log_dict["train"].append({
        "epoch": epoch, "loss": avg_loss, "token_accuracy": avg_acc
    })
    loss_history.append(avg_loss)

    if scheduler:
        scheduler.step()

    if epoch == 1 or avg_loss < best_loss:
        best_loss = avg_loss
        patience = 0
        torch.save({
            "paired_model": paired_model.state_dict(),
            "unpaired_model": unpaired_model.state_dict()
        }, SAVE_DIR / "best_model.pth")
        print("🧠 Best model saved!")
    else:
        patience += 1
        if patience >= PATIENCE_THRESHOLD:
            print(f"🛑 Early stopping at epoch {epoch}")
            break

# ✅ 최종 저장
save_log(log_dict, save_path=SAVE_ROOT / "train_log.json")
plot_loss_curve(loss_history, save_path=SAVE_ROOT / "loss_curve.png")
torch.save({
    "paired_model": paired_model.state_dict(),
    "unpaired_model": unpaired_model.state_dict()
}, SAVE_DIR / "final_model.pth")
print("✅ 최종 모델 저장 완료!")
