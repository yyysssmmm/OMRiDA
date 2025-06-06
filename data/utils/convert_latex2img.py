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

def render_latex_to_png(latex: str, output_path: Path, tmp_dir: Path, index: int, total: int):
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

        output_path.parent.mkdir(parents=True, exist_ok=True)
        (tmp_dir / output_path.name).rename(output_path)
        percent = (index + 1) / total * 100
        print(f"✅ [{index+1}/{total}] {output_path.name} 렌더링 성공 ({percent:.1f}%)")

    except Exception as e:
        percent = (index + 1) / total * 100
        print(f"⚠️ [{index+1}/{total}] {output_path.name} 렌더링 실패 ({percent:.1f}%): {e}")

def process_caption_file(caption_path: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir.parent / "tmp_render"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    with open(caption_path, "r", encoding="utf-8") as f:
        lines = [line for line in f if '\t' in line]

    total = len(lines)
    for idx, line in enumerate(lines):
        file_id, latex = line.strip().split('\t')
        output_path = output_dir / f"{file_id}.png"
        render_latex_to_png(latex, output_path, tmp_dir, idx, total)

    print(f"\n✅ 총 {total}개 수식 렌더링 시도 완료 → {output_dir}")

    # 임시 폴더 정리
    for f in tmp_dir.glob("*"):
        f.unlink()
    tmp_dir.rmdir()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=str, help="CROHME 연도 지정 예: 2014, 2019 등")
    parser.add_argument("--split", type=str, choices=["hme", "unpaired"], help="IM2LATEX split 지정: hme 또는 unpaired")
    args = parser.parse_args()

    if args.year:
        base_dir = Path(__file__).parent.parent / "CROHME" / "data_crohme"
        caption_path = base_dir / args.year / "caption.txt"
        output_dir = base_dir / args.year / "pme_img"
    
    elif args.split:
        base_dir = Path(__file__).parent.parent / "IM2LATEX"
        caption_path = base_dir / "caption" / f"{args.split}_caption.txt"
        output_dir = base_dir / "img" / f"{args.split}"

    else:
        raise ValueError("⚠️ '--year' 또는 '--split' 인자 중 하나는 반드시 지정해야 합니다.")

    process_caption_file(caption_path, output_dir)
