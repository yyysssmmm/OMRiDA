import torch
import editdistance  # pip install editdistance

def compute_exprate(preds, gts, pad_token_id=-1):
    """
    preds, gts: list of torch.Tensor (shape: (T,))
    returns: dict with 4 metrics (%)
    """
    total = len(preds)
    exact = count_1 = count_2 = count_3 = 0

    for pred, gt in zip(preds, gts):
        # padding 제거
        pred_tokens = pred[pred != pad_token_id].tolist()
        gt_tokens = gt[gt != pad_token_id].tolist()

        dist = editdistance.eval(pred_tokens, gt_tokens)

        if dist == 0:
            exact += 1
        if dist <= 1:
            count_1 += 1
        if dist <= 2:
            count_2 += 1
        if dist <= 3:
            count_3 += 1

    return {
        "ExpRate": 100 * exact / total,
        "<=1":     100 * count_1 / total,
        "<=2":     100 * count_2 / total,
        "<=3":     100 * count_3 / total,
    }
