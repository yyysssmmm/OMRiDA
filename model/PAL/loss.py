import torch
import torch.nn.functional as F

# Discriminator Loss
def compute_L_D(discriminator, a_xp, a_xh):
    """
    Adversarial loss for discriminator
    a_xp: (B, T, feature_dim) - printed
    a_xh: (B, T, feature_dim) - handwritten
    """
    B, T, D = a_xp.size()
    a_xp = a_xp.view(B * T, D)
    a_xh = a_xh.view(B * T, D)

    pred_real = discriminator(a_xp)  # printed → label 1
    pred_fake = discriminator(a_xh)  # handwritten → label 0

    loss_real = F.binary_cross_entropy(pred_real, torch.ones_like(pred_real))
    loss_fake = F.binary_cross_entropy(pred_fake, torch.zeros_like(pred_fake))
    return loss_real + loss_fake

# Cross Entropy Loss
def compute_L_C(a, y_true, classifier):
    """
    Cross entropy classification loss for a sequence
    a: (B, T, D) - feature sequences
    y_true: (B, T) - label indices
    classifier: module that maps feature vector to logits
    """
    B, T, D = a.size()
    logits = classifier(a.view(B * T, D))  # (B*T, num_classes)
    y_true = y_true.view(B * T)
    return F.cross_entropy(logits, y_true)

# Overall Loss
def compute_L_E(a_xp, a_xh, y_xp, y_xh, classifier, discriminator, lambda_adv):
    """
    Overall encoder loss L_E = L_Cp + L_Ch + lambda * L_D
    """
    L_Cp = compute_L_C(a_xp, y_xp, classifier)
    L_Ch = compute_L_C(a_xh, y_xh, classifier)

    B, T, D = a_xp.size()
    a_xp = a_xp.view(B * T, D)
    a_xh = a_xh.view(B * T, D)

    pred_xp = discriminator(a_xp)
    pred_xh = discriminator(a_xh)

    L_D = F.binary_cross_entropy(pred_xp, torch.zeros_like(pred_xp)) + \
          F.binary_cross_entropy(pred_xh, torch.ones_like(pred_xh))

    return L_Cp + L_Ch + lambda_adv * L_D