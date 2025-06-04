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
                context_h=None, context_p=None    # (B, T, C), (B, T, C)
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
            loss_h = self.ce_loss(logits_h.view(B * T, V), targets_h.view(B * T))
            loss_total += loss_h

        # Decoder Loss - Paired Printed
        if logits_p is not None and targets_p is not None:
            B, T, V = logits_p.size()
            loss_p = self.ce_loss(logits_p.view(B * T, V), targets_p.view(B * T))
            loss_total += loss_p

        # Decoder Loss - Unpaired Printed (optional)
        if logits_up is not None and targets_up is not None:
            B, T, V = logits_up.size()
            loss_up = self.ce_loss(logits_up.view(B * T, V), targets_up.view(B * T))
            loss_total += loss_up

        # Matching Loss - Context alignment
        if context_h is not None and context_p is not None:
            match_loss = self.mse_loss(context_h, context_p)
            loss_total += self.match_weight * match_loss
        else:
            match_loss = torch.tensor(0.0, device=logits_h.device if logits_h is not None else 'cpu')

        return loss_total, {
            "loss_h": loss_h.item() if logits_h is not None else 0.0,
            "loss_p": loss_p.item() if logits_p is not None else 0.0,
            "loss_up": loss_up.item() if logits_up is not None else 0.0,
            "match_loss": match_loss.item()
        }
