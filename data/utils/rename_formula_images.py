from pathlib import Path
import shutil

def rename_images(lst_path: Path, image_dir: Path):
    with open(lst_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    renamed = 0
    for line in lines:
        parts = line.split()
        if len(parts) < 2:
            continue
        idx, old_name = parts[0], parts[1]
        old_file = image_dir / f"{old_name}.png"
        new_file = image_dir / f"{idx}.png"

        if old_file.exists():
            old_file.rename(new_file)
            renamed += 1
        else:
            print(f"⚠️ 파일 없음: {old_file.name}")

    print(f"✅ {renamed}개 이미지 리네이밍 완료 (→ {image_dir})")

if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent / "IM2LATEX"
    lst_dir = base_dir / "raw_caption"
    img_dir = base_dir / "img" / "whole_img"

    # 필요한 lst 파일들
    lst_files = ["im2latex_train.lst", "im2latex_test.lst", "im2latex_validate.lst"]
    
    for lst_name in lst_files:
        lst_path = lst_dir / lst_name
        print(f"🔄 리네이밍 시작: {lst_path.name}")
        rename_images(lst_path, img_dir)
