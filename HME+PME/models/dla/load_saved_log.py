import yaml
import argparse


# ✅ config.yaml 불러오기
def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="config.yaml")
args = parser.parse_args()
config = load_config(args.config)


BATCH_LOG_PATH = config["misc"]["batch_log_path"]

import torch

# 저장된 로그 파일 경로
file_path = BATCH_LOG_PATH

# 로그 파일 로드
log = torch.load(file_path)

# 키 목록 확인
print("📦 저장된 키 목록:", log.keys())

# 원하는 값 출력
print("🔍 Loss:", log["loss"])
print("🔍 Token Accuracy:", log["token_accuracy"])
print("🔍 전체 Loss dict:", log["loss_dict"])
