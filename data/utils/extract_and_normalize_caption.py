from pathlib import Path
import re
from collections import Counter

def save_caption_file(path: Path, caption_pairs: list[tuple[int, str]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="latin1", newline="\n") as f:
        for idx, latex in caption_pairs:
            f.write(f"{idx}.png\t{latex}\n")

def extract_unpaired_latex_preserve_duplicates(base_dir: Path):
    # ✅ 경로 정의
    output_dir = base_dir / "caption"
    output_dir.mkdir(parents=True, exist_ok=True)

    tok_lst_path = base_dir / "im2latex_formulas.tok.lst"

    # ✅ 수식 리스트 정제
    with open(base_dir / "whole_im2latex.lst", "r", encoding="latin1", newline='\n') as f:
        whole_list = [re.sub(r"\s+", " ", line).strip() for line in f if line.strip()]

    with open(base_dir / "hme_im2latex.lst", "r", encoding="latin1", newline='\n') as f:
        paired_list = [re.sub(r"\s+", " ", line).strip() for line in f if line.strip()]

    with open(tok_lst_path, "r", encoding="latin1", newline='\n') as f:
        tok_lines = [line.strip() for line in f]

    assert len(whole_list) == len(tok_lines), (
        f"❌ whole_im2latex.lst ({len(whole_list)}) 와 im2latex_formula.tok.lst ({len(tok_lines)}) 길이가 다릅니다!"
    )

    paired_counter = Counter(paired_list)

    # ✅ 저장용 리스트
    whole_caption = []
    hme_caption = []
    unpaired_caption = []

    for idx, formula in enumerate(whole_list):
        tok_formula = tok_lines[idx]  # ✅ 인덱스 보장됨

        whole_caption.append((idx, tok_formula))

        if paired_counter[formula] > 0:
            paired_counter[formula] -= 1
            hme_caption.append((idx, tok_formula))
        else:
            unpaired_caption.append((idx, tok_formula))

    # ✅ caption 저장
    save_caption_file(output_dir / "whole_caption.txt", whole_caption)
    save_caption_file(output_dir / "hme_caption.txt", hme_caption)
    save_caption_file(output_dir / "unpaired_caption.txt", unpaired_caption)

    # ✅ 출력
    print("🔢 수식 개수 비교")
    print(" - 전체 수식 개수       :", len(whole_list))
    print(" - paired 수식 개수     :", len(paired_list))
    print("🧾 hme_caption 수식 수   :", len(hme_caption))
    print("🧾 unpaired_caption 수식 수:", len(unpaired_caption))
    print("✅ caption 파일 저장 완료 →", output_dir)

if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent / "IM2LATEX"
    extract_unpaired_latex_preserve_duplicates(base_dir)
