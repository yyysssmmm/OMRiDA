import os
import re
import shutil
import numpy as np
from tqdm import tqdm

# # ===== 경로 설정 =====
# base_dir = r"C:/Users/sophi/OneDrive/바탕 화면/Folder/2025 HYU/인공지능프로젝트/data/train/im2latex"
# caption_path = os.path.join(base_dir, "caption.txt")
# out_train = os.path.join(base_dir, "split_train.txt")
# out_test  = os.path.join(base_dir, "split_test.txt")

# train_ratio = 0.7
# seed = 42

# # ===== 유틸 =====
# def normalize_caption(text: str) -> str:
#     """캡션 공백 정규화 (동일 식 그룹핑 안정화)"""
#     text = (text or "").strip()
#     return " ".join(re.split(r"\s+", text)) if text else ""

# def token_len(text: str) -> int:
#     """공백 기준 토큰 길이"""
#     if not text:
#         return 0
#     return len(re.split(r"\s+", text.strip()))

# def write_split(lines, path):
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     with open(path, "w", encoding="utf-8") as f:
#         for filename, caption in lines:
#             cap = " ".join((caption or "").split())  # 저장 시도 공백 1칸
#             f.write(f"{filename} {cap}\n")

# # ===== 1) caption.txt 로드 =====
# entries = []
# with open(caption_path, "r", encoding="utf-8") as f:
#     for line in f:
#         line = line.strip()
#         if not line or line.startswith("#"):
#             continue
#         parts = line.split(None, 1)
#         if len(parts) == 1:
#             filename, caption = parts[0], ""
#         else:
#             filename, caption = parts[0], parts[1]
#         cap_norm = normalize_caption(caption)
#         entries.append((filename, caption, cap_norm))

# # ===== 2) 그룹핑 =====
# from collections import defaultdict
# group_to_items = defaultdict(list)
# for fname, cap, cap_norm in entries:
#     group_to_items[cap_norm].append((fname, cap))

# # 그룹별 대표 길이 계산
# groups = list(group_to_items.keys())
# lengths = np.array([token_len(groups[i]) for i in range(len(groups))], dtype=int)

# # ===== 3) 사분위 bin 만들기 =====
# def assign_quartile_bins_via_quantiles(lengths: np.ndarray):
#     if len(lengths) == 0:
#         return None
#     q1, q2, q3 = np.quantile(lengths, [0.25, 0.5, 0.75])
#     bins = []
#     for L in lengths:
#         if L <= q1:
#             bins.append(0)
#         elif L <= q2:
#             bins.append(1)
#         elif L <= q3:
#             bins.append(2)
#         else:
#             bins.append(3)
#     counts = [0, 0, 0, 0]
#     for b in bins: counts[b] += 1
#     if any(c == 0 for c in counts):
#         return None
#     return np.array(bins, dtype=int)

# def assign_quartile_bins_balanced(lengths: np.ndarray):
#     idx_sorted = np.argsort(lengths, kind="mergesort")
#     n = len(lengths)
#     splits = np.linspace(0, n, 5, dtype=int)
#     bins = np.empty(n, dtype=int)
#     for b in range(4):
#         start, end = splits[b], splits[b+1]
#         bins[idx_sorted[start:end]] = b
#     return bins

# bins = assign_quartile_bins_via_quantiles(lengths)

# # ===== 4) bin별로 그룹키 70% 랜덤 선택 =====
# rng = np.random.default_rng(seed)
# train_groups = set()
# test_groups = set()

# for b in range(4):
#     idx_in_bin = np.where(bins == b)[0]
#     keys_in_bin = [groups[i] for i in idx_in_bin]
#     n = len(keys_in_bin)
#     if n == 0:
#         continue
#     n_train = int(np.floor(n * train_ratio))
#     perm = rng.permutation(n)
#     sel_train = [keys_in_bin[i] for i in perm[:n_train]]
#     sel_test  = [keys_in_bin[i] for i in perm[n_train:]]
#     train_groups.update(sel_train)
#     test_groups.update(sel_test)

# # 누수 방지 체크
# assert train_groups.isdisjoint(test_groups), "그룹 누수 감지"

# # ===== 5) 라인 매핑 후 저장 =====
# train_lines = []
# test_lines = []
# for g in tqdm(groups, desc="Assigning groups to splits"):
#     items = group_to_items[g]  # [(filename, caption), ...]
#     if g in train_groups:
#         train_lines.extend(items)
#     else:
#         test_lines.extend(items)

# print(f"총 이미지: {len(entries)} | Train: {len(train_lines)} | Test: {len(test_lines)}")
# print(f"총 그룹(수식) 수: {len(groups)} | Train groups: {len(train_groups)} | Test groups: {len(test_groups)}")

# write_split(train_lines, out_train)
# write_split(test_lines, out_test)

# print(f"Saved:\n - {out_train}\n - {out_test}")

# ===== 경로 설정 =====
base_dir = r"C:/Users/sophi/OneDrive/바탕 화면/Folder/2025 HYU/인공지능프로젝트/data/train/im2latex"
split_train = os.path.join(base_dir, "split_train.txt")
split_test  = os.path.join(base_dir, "split_test.txt")

# 이미지 소스 폴더: 리사이즈된 폴더로 쓰는 걸 권장
src_dir = r"C:/Users/sophi/OneDrive/바탕 화면/Folder/2025 HYU/인공지능프로젝트/data/train/im2latex/pme"
dst_train_dir = os.path.join(base_dir, "images_train")
dst_test_dir  = os.path.join(base_dir, "images_test")
os.makedirs(dst_train_dir, exist_ok=True)
os.makedirs(dst_test_dir, exist_ok=True)

def iter_filenames(split_path):
    with open(split_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            fname = line.split(None, 1)[0]
            yield fname

def link_or_copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)

def materialize_split(split_path, src_root, dst_root, label="train"):
    for fname in tqdm(iter_filenames(split_path), desc=f"Making {label} set"):
        src = os.path.normpath(os.path.join(src_root, fname))
        dst = os.path.normpath(os.path.join(dst_root, fname))
        if not os.path.exists(src):
            print(f"[WARN] Missing: {src}")
            continue
        link_or_copy(src, dst)

# ===== 실행 =====
materialize_split(split_train, src_dir, dst_train_dir, label="train")
materialize_split(split_test,  src_dir, dst_test_dir,  label="test")
print("Done.")