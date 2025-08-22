#  1) vocab.txt가 caption.txt 토큰을 얼마나 커버하는지 (token-level %) 출력
#  2) vocab.txt에는 없고 caption.txt에는 등장하는 모든 토큰을 출력

from pathlib import Path
import os
import re
import collections

# 경로 설정
base_dir = "C:/Users/sophi/OneDrive/바탕 화면/Folder/2025 HYU/인공지능프로젝트/data/crohme"
CAPTION_PATH = os.path.join(base_dir, "caption.txt")
VOCAB_PATH   = os.path.join(base_dir, "crohme_vocab.txt")

# 아주 단순한 LaTeX 토크나이저 (공백 없을 때만 사용)
LATEX_TOKEN_RE = re.compile(
    r"""
    (\\[a-zA-Z]+)            |  # \command
    (\\.)                    |  # \{ 같은 1글자 이스케이프
    ([{}\[\]\(\)\^\_\&\%\#\~\$]) |  # 단일 특수기호
    ([0-9]+)                 |  # 숫자 시퀀스
    ([a-zA-Z]+)              |  # 영문 시퀀스
    (\S)                        # 기타 한 글자
    """,
    re.VERBOSE,
)

def tokenize(line: str):
    s = line.strip()
    if not s:
        return []
    # 공백 분리 흔적이 있으면 그대로 사용
    if " " in s:
        return s.split()
    # 아니면 간단 LaTeX 규칙으로 분리
    toks = []
    for m in LATEX_TOKEN_RE.finditer(s):
        toks.append(next(g for g in m.groups() if g is not None))
    return toks

vocab = set()
for ln in Path(VOCAB_PATH).read_text(encoding="utf-8").splitlines():
    t = ln.strip()
    if t:
        vocab.add(t)

total_tokens = 0
oov_tokens = 0
oov_counter = collections.Counter()

for ln in Path(CAPTION_PATH).read_text(encoding="utf-8").splitlines():
    parts = ln.strip().split(maxsplit=1)
    if len(parts) < 2:
        continue  # 캡션 없는 경우 스킵
    caption = parts[1]
    toks = tokenize(caption)
    total_tokens += len(toks)
    for t in toks:
        if t not in vocab:
            oov_tokens += 1
            oov_counter[t] += 1

# 1) 토큰 단위 커버리지(%)
coverage = 100.0 * (1 - (oov_tokens / max(1, total_tokens)))
print(f"Token coverage: {coverage:.2f}% "
        f"(total={total_tokens}, OOV={oov_tokens}, types={len(oov_counter)})")

# 2) vocab에 없는 토큰 전체
print(f"Top 30 OOV tokens:")
for tok, cnt in oov_counter.most_common(30):
    print(f"{tok}\t{cnt}")