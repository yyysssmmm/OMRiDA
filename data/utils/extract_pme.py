from pathlib import Path
import shutil
import argparse

def extract_unpaired_images(split):
    # 경로 설정
    base_dir = Path(__file__).parent.parent / "IM2LATEX"
    if split=="paired":
        caption_path = base_dir / "caption" / "hme_caption.txt"
    else:
        caption_path = base_dir / "caption" / "unpaired_caption.txt"

    source_dir = base_dir / "img" / "whole_img"
    target_dir = base_dir / "img" / f"pme_{split}"
    target_dir.mkdir(parents=True, exist_ok=True)

    # 파일명 추출
    with open(caption_path, "r", encoding="latin1") as f:
        lines = [line.strip() for line in f if line.strip()]
        file_names = [line.split('\t')[0] for line in lines if '\t' in line]

    copied = 0
    for file_name in file_names:
        src_path = source_dir / file_name
        dst_path = target_dir / file_name
        if src_path.exists():
            shutil.copy(src_path, dst_path)
            copied += 1
        else:
            print(f"⚠️ 존재하지 않는 파일: {file_name}")

    print(f"✅ 총 {copied}개 이미지 복사 완료 → {target_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, choices=["paired", "unpaired"], help="IM2LATEX split 지정: paired 또는 unpaired")
    args = parser.parse_args()

    extract_unpaired_images(split=args.split)
