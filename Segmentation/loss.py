import torch
import torch.nn.functional as F
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)
        intersection = (preds * targets).sum()
        union = preds.sum() + targets.sum()
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice


class BCEWithLogitsLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, preds, targets):
        return self.bce(preds, targets)


# pixel-level contrastive loss
def pixel_contrastive_loss(f1, f2, label, temperature=0.1):
    """
    f1, f2: encoder feature maps (B, C, H, W)
    label: batch-level pair label: (B,), 0 for positive pair, 1 for negative pair
    """
    B, C, H, W = f1.shape

    f1 = F.normalize(f1, dim=1)
    f2 = F.normalize(f2, dim=1)

    f1_flat = f1.permute(0,2,3,1).reshape(B, -1, C)  # (B, HW, C)
    f2_flat = f2.permute(0,2,3,1).reshape(B, -1, C)

    sim = (f1_flat * f2_flat).sum(dim=2) / temperature  # (B, HW)

    pos_mask = (label == 0).float().unsqueeze(1)
    neg_mask = (label == 1).float().unsqueeze(1)

    pos_loss = -(torch.log(torch.sigmoid(sim)) * pos_mask).sum() / (pos_mask.sum() + 1e-6)
    neg_loss = -(torch.log(1 - torch.sigmoid(sim)) * neg_mask).sum() / (neg_mask.sum() + 1e-6)

    return pos_loss + neg_loss
    