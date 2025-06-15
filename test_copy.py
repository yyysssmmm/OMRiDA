import os
import json
import yaml
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from models.dla import DLAModel
from data.dataset import FormulaDataset
from data.utils.vocab import Vocab
from utils.metrics import exprate_k, cer
from utils.decode import decode_sequence
from utils.seed import set_seed
from utils.data_utils import get_formula_transform, formula_collate_fn

# ✅ config.yaml 불러오기
def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="config.yaml")
args = parser.parse_args()
config = load_config(args.config)

# ✅ 기본 설정값 로드
set_seed(config["misc"]["seed"])
IGNORE_IDX = config["training"]["ignore_idx"]
BATCH_SIZE = config["testing"].get("batch_size", 1)
MAX_LEN = config["testing"].get("max_len", 150)
TEST_YEARS = ["2014", "2016", "2019"]
CHECKPOINT_PATH = config["misc"]["batch_log_path"]
VOCAB_PATH = config["data"]["vocab"]
DEVICE = torch.device(config["misc"]["device"] if torch.backends.mps.is_available() or torch.cuda.is_available() else "cpu")

# ✅ 결과 디렉토리
os.makedirs("preds", exist_ok=True)

# ✅ vocab 및 모델 로드
vocab = Vocab.load_from_txt(Path(VOCAB_PATH))
SOS_ID = vocab.token2idx["<sos>"]
EOS_ID = vocab.token2idx["<eos>"]

model_config = config["model"]
unpaired_model = DLAModel(vocab_size=len(vocab), model_config=model_config).to(DEVICE)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
unpaired_model.load_state_dict(checkpoint["model_state_dict"])
unpaired_model.eval()

# ✅ 평가 함수 정의
def evaluate(img_dir, caption_path, img_ext, mode="hme"):
    assert mode in ["hme", "pme"], f"❌ Invalid mode: {mode}"

    transform = get_formula_transform("paired_hme" if mode == "hme" else "paired_pme", config["transforms"])
    dataset = FormulaDataset(
        image_dir=img_dir,
        caption_path=caption_path,
        transform=transform,
        image_ext=img_ext,
        vocab=vocab
    )
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda b: formula_collate_fn(b, pad_idx=IGNORE_IDX)
    )

    preds, targets = [], []
    debug_lines = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"{mode.upper()} Eval"):
            imgs = batch["image"].to(DEVICE)
            tgt_ids = batch["formula"]

            pred_ids_batch = unpaired_model.predict(imgs, max_len=MAX_LEN, sos_idx=SOS_ID, eos_idx=EOS_ID)
            pred_tokens_batch = decode_sequence(pred_ids_batch, vocab)
            target_tokens_batch = decode_sequence(tgt_ids, vocab)

            preds.extend(pred_tokens_batch)
            targets.extend(target_tokens_batch)

            for pred, target in zip(pred_tokens_batch, target_tokens_batch):
                debug_lines.append(f"[GT]   {' '.join(target)}")
                debug_lines.append(f"[PRD]  {' '.join(pred)}")
                debug_lines.append("---")

    year = Path(img_dir).parts[-2]
    debug_path = f"preds/pred_target_pairs_{mode}_{year}.txt"
    with open(debug_path, "w", encoding="latin1") as f:
        f.write("\n".join(debug_lines))
    print(f"📄 {mode.upper()} {year} 결과 저장됨: {debug_path}")

    return preds, targets

# ✅ 결과 출력 함수
def print_result_section(title: str, result_dict: dict):
    print(f"\n📊 {title}")
    for key, val in result_dict.items():
        print(f"  {key}: {val}")

# ✅ 전체 평가
result_log = {"CROHME": {"hme": {}, "pme": {}}, "IM2LATEX": {"pme":{}}}

# hme 평가
for year in TEST_YEARS:
    img_dir = f"data/preprocessed/test/hme/crohme/{year}/hme_img"
    caption_path = f"data/preprocessed/test/hme/crohme/{year}/caption.txt"
    preds, targets = evaluate(img_dir, caption_path, img_ext="bmp", mode="hme")

    result_log["CROHME"]["hme"][year] = {
        f"exprate_{k}": round(exprate_k(preds, targets, k), 4) for k in range(4)
    }
    result_log["CROHME"]["hme"][year]["cer"] = round(cer(preds, targets), 4)

# pme 평가
for year in TEST_YEARS:
    img_dir = f"data/preprocessed/test/pme/crohme/{year}/pme_img"
    caption_path = f"data/preprocessed/test/pme/crohme/{year}/caption.txt"
    preds, targets = evaluate(img_dir, caption_path, img_ext="png", mode="pme")

    result_log["CROHME"]["pme"][year] = {
        f"exprate_{k}": round(exprate_k(preds, targets, k), 4) for k in range(4)
    }
    result_log["CROHME"]["pme"][year]["cer"] = round(cer(preds, targets), 4)

# im2latex 평가
img_dir = f"data/preprocessed/test/pme/im2latex/img"
caption_path = f"data/preprocessed/test/pme/im2latex/caption.txt"
preds, targets = evaluate(img_dir, caption_path, img_ext="png", mode="pme")

result_log["IM2LATEX"]["pme"] = {
    f"exprate_{k}": round(exprate_k(preds, targets, k), 4) for k in range(4)
}
result_log["IM2LATEX"]["pme"]["cer"] = round(cer(preds, targets), 4)

# ✅ 출력
for year in TEST_YEARS:
    print_result_section(f"CROHME-HME {year}", result_log["CROHME"]["hme"][year])
    print_result_section(f"CROHME-PME {year}", result_log["CROHME"]["pme"][year])

print_result_section("IM2LATEX-PME", result_log["IM2LATEX"]["pme"])

# ✅ 전체 결과 저장
with open("test_results.json", "w") as f:
    json.dump(result_log, f, indent=4)

print("\n✅ 평가 완료! 결과는 test_results.json에 저장됨")
