import torch
import numpy as np
from difflib import SequenceMatcher


def token_accuracy(preds, targets, pad_token=0):
    """
    Token-level accuracy (ignoring pad_token).
    """
    mask = targets != pad_token
    correct = (preds == targets) & mask
    accuracy = correct.sum().item() / mask.sum().item()
    return accuracy


def exprate_k(preds, targets, k):
    """
    Expression rate-k: 예측 수식과 정답 수식이 k개 이하의 토큰 차이만 있을 때 정답으로 간주.
    길이 차이도 diff에 포함하여 평가.
    """
    assert len(preds) == len(targets)
    correct = 0
    for p, t in zip(preds, targets):
        diff = abs(len(p) - len(t))
        for a, b in zip(p, t):
            if a != b:
                diff += 1
        if diff <= k:
            correct += 1
    return correct / len(preds)


def cer(preds, targets):
    """
    Character Error Rate (token-level edit distance / length of target).
    평균 CER을 전체 샘플에 대해 계산.
    """
    total_distance = 0
    total_length = 0
    for p, t in zip(preds, targets):
        total_distance += levenshtein_distance(p, t)
        total_length += len(t)
    return total_distance / total_length if total_length > 0 else 0



def levenshtein_distance(seq1, seq2):
    """
    기본적인 편집 거리 계산 함수 (DP 기반).
    """
    n, m = len(seq1), len(seq2)
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost  # substitution
            )
    return dp[n][m]