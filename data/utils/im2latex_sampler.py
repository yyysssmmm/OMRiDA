from pathlib import Path
import os
import shutil

def load_vocab(vocab_path):
    with open(vocab_path, 'r', encoding='latin1') as f:
        vocab = set(token.strip() for token in f)
    return vocab

def is_valid_formula(formula: str, vocab: set):
    tokens = formula.strip().split()
    return all(tok in vocab for tok in tokens)

def sample_captions_and_copy_images(caption_path, vocab, save_caption_path, img_src_dir, img_dst_dir):
    valid_lines = []
    copied = 0

    # 이미지 저장 디렉토리 생성
    os.makedirs(img_dst_dir, exist_ok=True)

    with open(caption_path, 'r', encoding='latin1') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            img_name, formula = parts

            if is_valid_formula(formula, vocab):
                valid_lines.append(line.strip())

                # 이미지 복사
                src_path = img_src_dir / img_name
                dst_path = img_dst_dir / img_name

                if src_path.exists():
                    shutil.copyfile(src_path, dst_path)
                    copied += 1
                else:
                    print(f"⚠️ 이미지 누락: {src_path}")

    # 유효한 caption 저장
    with open(save_caption_path, 'w', encoding='latin1') as f:
        for line in valid_lines:
            f.write(line + '\n')

    print(f"✅ {len(valid_lines)}개 수식 저장됨: {os.path.basename(save_caption_path)}")
    print(f"📦 {copied}개 이미지 복사 완료 → {img_dst_dir}")


if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent
    vocab_path = base_dir / "vocab/crohme_vocab.txt"
    vocab = load_vocab(vocab_path)

    # caption 파일 경로
    caption_dir = base_dir / "IM2LATEX/caption"
    caption_file_paired = caption_dir / "hme_caption.txt"
    caption_file_unpaired = caption_dir / "unpaired_caption.txt"

    # 이미지 source 디렉토리
    img_dir = base_dir / "IM2LATEX/img"
    img_pme_paired = img_dir / "pme_paired"
    img_hme_paired = img_dir / "hme_paired"
    img_pme_unpaired = img_dir / "pme_unpaired"

    # 저장할 caption 위치
    save_caption_unpaired = base_dir / "preprocessed/test/pme/im2latex/caption.txt"
    save_caption_paired = base_dir / "preprocessed/train/paired/im2latex/caption.txt"

    # 저장할 이미지 폴더
    save_unpaired_img = save_caption_unpaired.parent / "img"
    save_paired_img_hme = save_caption_paired.parent  / "hme"
    save_paired_img_pme = save_caption_paired.parent  / "pme"

    # 디렉토리 생성 (caption.txt는 파일이므로 .parent로 접근)
    os.makedirs(save_caption_unpaired.parent, exist_ok=True)
    os.makedirs(save_caption_paired.parent, exist_ok=True)

    # 실행
    sample_captions_and_copy_images(
        caption_file_unpaired, vocab, save_caption_unpaired, img_pme_unpaired, save_unpaired_img
    )
    sample_captions_and_copy_images(
        caption_file_paired, vocab, save_caption_paired, img_hme_paired, save_paired_img_hme
    )
    sample_captions_and_copy_images(
        caption_file_paired, vocab, save_caption_paired, img_pme_paired, save_paired_img_pme
    )
