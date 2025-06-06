from pathlib import Path
from typing import List
import re

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
            for tok in formula:
                if tok not in self.token2idx:
                    self.token2idx[tok] = len(self.idx2token)
                    self.idx2token.append(tok)

    def encode(self, formula: str) -> List[int]:
        tokens = formula.strip().split()
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
    def load_from_txt(cls, path: Path) -> "Vocab":   # ✅ 여기 안으로 넣기!
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
    vocab_dir = base_dir / "vocab"

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
    print(f" - Size: {len(crohme_vocab)}")
    print(f" - tokens: {sorted(crohme_vocab.idx2token)}\n")

    # ✅ IM2LATEX vocab
    caption_path = im2latex_caption_dir / "whole_caption.txt"
    im2latex_formulas = load_caption_formulas(caption_path)
    im2latex_vocab = Vocab()
    im2latex_vocab.build_vocab(im2latex_formulas)
    im2latex_vocab.save_to_txt(vocab_dir / "im2latex_vocab.txt")
    print("📗 IM2LATEX Vocabulary")
    print(f" - Size: {len(im2latex_vocab)}")
    print(f" - tokens: {sorted(im2latex_vocab.idx2token)}\n")
