import shutil
from pathlib import Path
from PIL import Image  # ✅ 이미지 변환용
import sys

# 📁 경로 설정
BASE_DIR = Path(__file__).resolve().parent.parent
CROHME_HME_DIR = BASE_DIR / "CROHME/data_crohme/train/img"
CROHME_PME_DIR = BASE_DIR / "CROHME/data_crohme/train/pme_img"
CROHME_CAPTION_PATH = BASE_DIR / "CROHME/data_crohme/train/caption.txt"

IM2LATEX_HME_DIR = BASE_DIR / "IM2LATEX/img/hme_paired"
IM2LATEX_PME_DIR = BASE_DIR / "IM2LATEX/img/pme_paired"
IM2LATEX_CAPTION_PATH = BASE_DIR / "IM2LATEX/caption/hme_caption.txt"

OUT_DIR = BASE_DIR / "paired_CROHME+IM2LATEX"
OUT_HME = OUT_DIR / "hme"
OUT_PME = OUT_DIR / "pme"
OUT_CAPTION = OUT_DIR / "caption.txt"

# 🔧 출력 폴더 초기화
for folder in [OUT_HME, OUT_PME]:
    folder.mkdir(parents=True, exist_ok=True)

with open(OUT_CAPTION, "w", encoding="latin1") as out_f:
    idx = 0

    # ✅ CROHME 처리
    with open(CROHME_CAPTION_PATH, "r", encoding="latin1", newline='\n') as f:
        for line in f:
            img_name, formula = line.strip().split('\t')
            new_name = f"crohme_{idx:05d}.png"

            hme_src = CROHME_HME_DIR / f"{img_name}.bmp"
            pme_src = CROHME_PME_DIR / f"{img_name}.png"

            hme_dst = OUT_HME / new_name
            pme_dst = OUT_PME / new_name

            if hme_src.exists() and pme_src.exists():
                try:
                    # ✅ BMP → PNG 변환
                    img = Image.open(hme_src)
                    img.save(hme_dst, format="PNG")
                except Exception as e:
                    print(f"[오류] HME 변환 실패: {hme_src} → {e}", file=sys.stderr)
                    continue

                shutil.copy(pme_src, pme_dst)
                out_f.write(f"{new_name}\t{formula.strip()}\n")
                idx += 1

    # ✅ IM2LATEX 처리
    with open(IM2LATEX_CAPTION_PATH, "r", encoding="latin1", newline='\n') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue

            img_name, formula = parts
            new_name = f"im2latex_{idx:05d}.png"

            hme_src = IM2LATEX_HME_DIR / img_name
            pme_src = IM2LATEX_PME_DIR / img_name

            hme_dst = OUT_HME / new_name
            pme_dst = OUT_PME / new_name

            if hme_src.exists() and pme_src.exists():
                shutil.copy(hme_src, hme_dst)
                shutil.copy(pme_src, pme_dst)
                out_f.write(f"{new_name}\t{formula.strip()}\n")
                idx += 1

print(f"✅ 통합 완료: 총 {idx} 쌍의 paired 데이터 저장됨.")
