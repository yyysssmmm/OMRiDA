def decode_sequence(token_ids, vocab):
    results = []
    for seq in token_ids:
        tokens = []
        for idx in seq:
            token = vocab.idx2token[int(idx)]
            if token == "<pad>":
                continue
            if token == "<eos>":
                break
            tokens.append(token)
        results.append(" ".join(tokens))
    return results
