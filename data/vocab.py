"""
vocab.py

공식 수식에서 사용하는 토큰들을 사전으로 관리하는 클래스 정의 파일.

- <pad>, <sos>, <eos> 등의 특수 토큰 자동 처리
- 수식 문자열을 토큰 시퀀스로 인코딩하는 기능 제공
- 디코딩(역변환)은 필요시 추가 가능
"""

class Vocab:
    PAD_TOKEN = "<pad>"
    SOS_TOKEN = "<sos>"
    EOS_TOKEN = "<eos>"

    def __init__(self):
        # 토큰 목록 초기화
        self.tokens = [self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN]
        self.token2idx = {tok: idx for idx, tok in enumerate(self.tokens)}
        self.idx2token = self.tokens.copy()

    def build_vocab(self, formula_list):
        """
        수식 리스트로부터 고유한 토큰을 추출하여 vocab 구성

        Args:
            formula_list: 수식 문자열 리스트 (공백 기준 토큰화)
        """
        for formula in formula_list:
            for tok in formula.strip().split():
                if tok not in self.token2idx:
                    self.token2idx[tok] = len(self.idx2token)
                    self.idx2token.append(tok)

    def encode(self, formula):
        """
        수식 문자열을 토큰 인덱스 시퀀스로 변환

        Args:
            formula: 수식 문자열

        Returns:
            List[int]: [<sos>, token_id1, token_id2, ..., <eos>]
        """
        tokens = formula.strip().split()
        return [self.token2idx[self.SOS_TOKEN]] + \
               [self.token2idx.get(tok, self.token2idx[self.PAD_TOKEN]) for tok in tokens] + \
               [self.token2idx[self.EOS_TOKEN]]

    def __len__(self):
        return len(self.idx2token)

    def decode(self, token_ids):
        """
        토큰 인덱스 시퀀스를 수식 문자열로 변환 (디버깅용)

        Args:
            token_ids: List[int]

        Returns:
            str: 공백 구분 수식 문자열
        """
        return ' '.join([self.idx2token[idx] for idx in token_ids if idx < len(self.idx2token)])


# ✅ 테스트 예시
if __name__ == "__main__":
    formula_list = [
        r"\frac { a } { b }",
        r"a + b = c",
        r"\sqrt { x ^ 2 + y ^ 2 }"
    ]

    vocab = Vocab()
    vocab.build_vocab(formula_list)

    print(f"📚 Vocabulary size: {len(vocab)}")
    for formula in formula_list:
        encoded = vocab.encode(formula)
        decoded = vocab.decode(encoded)
        print(f"🔢 {formula} → {encoded} → {decoded}")
