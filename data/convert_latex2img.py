import argparse
from pathlib import Path
import subprocess

TEX_TEMPLATE = r"""\documentclass[preview]{standalone}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{bm}
\begin{document}
${latex}
\end{document}
"""

def render_latex_to_png(latex: str, output_path: Path, tmp_dir: Path):
    tex_path = tmp_dir / "equation.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(TEX_TEMPLATE.replace("${latex}", f"${latex}$"))

    try:
        subprocess.run(["pdflatex", "-interaction=nonstopmode", tex_path.name],
                       cwd=tmp_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        subprocess.run(["pdfcrop", "equation.pdf", "equation-crop.pdf"],
                       cwd=tmp_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        subprocess.run(["convert", "-density", "200", "equation-crop.pdf", "-quality", "100", output_path.name],
                       cwd=tmp_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        # move result to final path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        (tmp_dir / output_path.name).rename(output_path)
    except Exception as e:
        print(f"⚠️ 렌더링 실패 ({output_path.stem}): {e}")

def process_caption_file(caption_path: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir.parent / "tmp_render"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    with open(caption_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    count = 0
    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) != 2:
            continue
        file_id, latex = parts
        output_path = output_dir / f"{file_id}.png"
        render_latex_to_png(latex, output_path, tmp_dir)
        count += 1

    print(f"✅ 총 {count}개 수식 렌더링 완료 → {output_dir}")

    # 임시 폴더 정리
    for f in tmp_dir.glob("*"):
        f.unlink()
    tmp_dir.rmdir()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=str, required=True)
    args = parser.parse_args()

    base_dir = Path(__file__).parent / "CROHME" / "data_crohme"
    caption_path = base_dir / args.year / "caption.txt"
    output_dir = base_dir / args.year / "pme_img"

    process_caption_file(caption_path, output_dir)
