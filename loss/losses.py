import torch
import torch.nn as nn
import torch.nn.functional as F

class DualLoss(nn.Module):
    """
    Dual loss function for paired dual loss attention model.

    Loss = LD(Xh) + LD(Xp) + LD(X̄) + λ * Lmatch(Xh, Xp)
    """
    def __init__(self, match_weight=1.0, ignore_index=0):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.mse_loss = nn.MSELoss()
        self.match_weight = match_weight

    def forward(self, 
                logits_h, targets_h,        # (B, T, V), (B, T)
                logits_p, targets_p,        # (B, T, V), (B, T)
                logits_up=None, targets_up=None,  # (B, T, V), (B, T)
                context_h=None, context_p=None    # (B, T, C),d (B, T, C)
               ):
        """
        logits_*: unnormalized decoder outputs (B, T, V)
        targets_*: target indices (B, T)
        context_*: attention context vectors (B, T, C)
        """
        loss_total = 0.0

        # Decoder Loss - Handwritten
        if logits_h is not None and targets_h is not None:
            B, T, V = logits_h.size()
            targets_h = targets_h[:, 1:]
            loss_h = self.ce_loss(logits_h.reshape(-1, V), targets_h.reshape(-1))
            loss_total += loss_h

        # Decoder Loss - Paired Printed
        if logits_p is not None and targets_p is not None:
            B, T, V = logits_p.size()
            targets_p = targets_p[:, 1:]
            loss_p = self.ce_loss(logits_p.reshape(-1, V), targets_p.reshape(-1))
            loss_total += loss_p

        # Decoder Loss - Unpaired PME
        if logits_up is not None and targets_up is not None:
            B, T, V = logits_up.size()
            targets_up = targets_up[:, 1:]
            loss_up = self.ce_loss(logits_up.reshape(-1, V), targets_up.reshape(-1))
            loss_total += loss_up

        else:
            loss_up = torch.tensor(0.0)

        # Context Matching Loss
        if context_h is not None and context_p is not None:
            assert context_h.shape == context_p.shape
            match_loss = self.mse_loss(context_h, context_p)
            loss_total += self.match_weight * match_loss
        else:
            match_loss = torch.tensor(0.0)

        return loss_total, {
            "loss_h": loss_h.item(),
            "loss_p": loss_p.item(),
            "loss_up": loss_up.item(),
            "match_loss": match_loss.item(),
            "total": loss_total.item()
        }
