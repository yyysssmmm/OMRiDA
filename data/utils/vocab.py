from pathlib import Path
from typing import List
import re


def tokenize_formula(formula):
    ans = []
    tokens = formula.strip().split()

    for tok in tokens:
        chk = 0
        for sym in ["cm", "mm", "pt", "in", "ex", "em"]:
            if (tok[-2:] == sym) and tok[-3].isdigit(): 
                num, _ = tok.split(sym)
                num = ' '.join(num).split()
                ans.extend(num)
                ans.append(sym)
                chk = 1
                break

        if chk == 0:
            ans.append(tok)
    return ans


class Vocab:
    PAD_TOKEN = "<pad>"
    SOS_TOKEN = "<sos>"
    EOS_TOKEN = "<eos>"

    def __init__(self):
        self.tokens = [self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN]
        self.token2idx = {tok: idx for idx, tok in enumerate(self.tokens)}
        self.idx2token = self.tokens.copy()

    def build_vocab(self, formula_list: List[str]):
        for formula in formula_list:
            tokenized_formula = tokenize_formula(formula)
            for tok in tokenized_formula:
                if tok not in self.token2idx:
                    self.token2idx[tok] = len(self.idx2token)
                    self.idx2token.append(tok)

    def encode(self, formula: str) -> List[int]:
        tokens = tokenize_formula(formula)
        return [self.token2idx[self.SOS_TOKEN]] + \
               [self.token2idx.get(tok, self.token2idx[self.PAD_TOKEN]) for tok in tokens] + \
               [self.token2idx[self.EOS_TOKEN]]

    def decode(self, token_ids: List[int]) -> str:
        return ' '.join([self.idx2token[idx] for idx in token_ids if idx < len(self.idx2token)])

    def __len__(self):
        return len(self.idx2token)

    def save_to_txt(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='latin1') as f:
            for token in sorted(self.idx2token):
                f.write(token + '\n')

    @classmethod
    def load_from_txt(cls, path: Path) -> "Vocab":
        vocab = cls()
        with open(path, "r", encoding="latin1") as f:
            for line in f:
                token = line.strip()
                if token not in vocab.token2idx:
                    vocab.token2idx[token] = len(vocab.idx2token)
                    vocab.idx2token.append(token)
        return vocab


def load_caption_formulas(caption_path: Path) -> List[str]:
    formulas = []
    with open(caption_path, "r", encoding="latin1") as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            _, formula = parts
            formulas.append(formula)
    return formulas


if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent

    # 📌 경로 설정
    crohme_dir = base_dir / "CROHME/data_crohme"
    im2latex_caption_dir = base_dir / "IM2LATEX/caption"
    paired_caption_path = base_dir / "paired_CROHME+IM2LATEX/caption.txt"
    unpaired_caption_path = base_dir / "IM2LATEX/caption/unpaired_caption.txt"
    paired_IM2LATEX_caption_path = base_dir / "IM2LATEX/caption/hme_caption.txt"
    vocab_dir = base_dir / "vocab"

    all_formulas = []

    # ✅ CROHME vocab
    crohme_formulas = []
    for year in ["2014", "2016", "2019", "train"]:
        caption_path = crohme_dir / year / "caption.txt"
        if caption_path.exists():
            crohme_formulas += load_caption_formulas(caption_path)

    crohme_vocab = Vocab()
    crohme_vocab.build_vocab(crohme_formulas)
    crohme_vocab.save_to_txt(vocab_dir / "crohme_vocab.txt")
    print("📘 CROHME Vocabulary")
    print(f" - Size: {len(crohme_vocab)}\n")
    # print(f" - tokens: {sorted(crohme_vocab.idx2token)}\n")
    all_formulas += crohme_formulas  

    # ✅ IM2LATEX vocab
    caption_path = im2latex_caption_dir / "whole_caption.txt"
    im2latex_formulas = load_caption_formulas(caption_path)
    im2latex_vocab = Vocab()
    im2latex_vocab.build_vocab(im2latex_formulas)
    im2latex_vocab.save_to_txt(vocab_dir / "im2latex_vocab.txt")
    print("📗 IM2LATEX Vocabulary")
    print(f" - Size: {len(im2latex_vocab)}\n")
    # print(f" - tokens: {sorted(im2latex_vocab.idx2token)}\n")
    all_formulas += im2latex_formulas  

    # ✅ PAIRED vocab (CROHME + IM2LATEX)
    if paired_caption_path.exists():
        paired_formulas = load_caption_formulas(paired_caption_path)
        paired_vocab = Vocab()
        paired_vocab.build_vocab(paired_formulas)
        paired_vocab.save_to_txt(vocab_dir / "paired_vocab.txt")
        print("📙 PAIRED (CROHME + IM2LATEX) Vocabulary")
        print(f" - Size: {len(paired_vocab)}\n")
        # print(f" - tokens: {sorted(paired_vocab.idx2token)}\n")
        all_formulas += paired_formulas 
    else:
        print("⚠️ PAIRED caption.txt가 존재하지 않습니다.\n")

    # ✅ IM2LATEX PAIRED vocab
    if paired_caption_path.exists():
        paired_formulas = load_caption_formulas(paired_IM2LATEX_caption_path)
        paired_vocab = Vocab()
        paired_vocab.build_vocab(paired_formulas)
        paired_vocab.save_to_txt(vocab_dir / "IM2LATEX_paired_vocab.txt")
        print("📙 PAIRED (IM2LATEX) Vocabulary")
        print(f" - Size: {len(paired_vocab)}\n")
        # print(f" - tokens: {sorted(paired_vocab.idx2token)}\n")
        all_formulas += paired_formulas 
    else:
        print("⚠️ IM2LATEX PAIRED caption.txt가 존재하지 않습니다.\n")

    # ✅ IM2LATEX UNPAIRED vocab
    if unpaired_caption_path.exists():
        unpaired_formulas = load_caption_formulas(unpaired_caption_path)
        unpaired_vocab = Vocab()
        unpaired_vocab.build_vocab(unpaired_formulas)
        unpaired_vocab.save_to_txt(vocab_dir / "im2latex_unpaired_vocab.txt")
        print("📒 IM2LATEX Unpaired Vocabulary")
        print(f" - Size: {len(unpaired_vocab)}\n")
        # print(f" - tokens: {sorted(unpaired_vocab.idx2token)}\n")
        all_formulas += unpaired_formulas  
    else:
        print("⚠️ unpaired_caption.txt가 존재하지 않습니다.\n")

    # ✅ 🔥 전체 통합 vocab (crohme + im2latex_paired + im2latex_unpaired + paired)
    all_vocab = Vocab()
    all_vocab.build_vocab(all_formulas)
    all_vocab.save_to_txt(vocab_dir / "all_vocab.txt")
    print("📕 ALL Vocabulary (전체 통합)")
    print(f" - Size: {len(all_vocab)}")
    # print(f" - tokens: {sorted(all_vocab.idx2token)}\n")
