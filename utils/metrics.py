import torch

def token_accuracy(preds, targets, pad_token=0):
    mask = targets != pad_token
    correct = (preds == targets) & mask
    accuracy = correct.sum().item() / mask.sum().item()
    return accuracy
