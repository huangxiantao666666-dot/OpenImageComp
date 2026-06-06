"""
Loss functions for TopNet training.

- ``CrossEntropyLoss`` (original):  sparse pixel classification, ignore_index=255.
- ``FocalLoss`` (CenterNet-style):  dense heatmap regression, α=2, β=4.
- ``get_loss(config)``: select loss by YAML ``loss`` field.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
#  Focal Loss  (CenterNet-style)
# ======================================================================
class FocalLoss(nn.Module):
    """
    Focal Loss for dense keypoint heatmap regression.

    Adapted from CenterNet (Objects as Points, Zhou et al. 2019):

        L_k = -1/N Σ (1-Ŷ)^α log(Ŷ)        for Y near 1  (Gaussian peaks)
                     (1-Y)^β Ŷ^α log(1-Ŷ)  for Y near 0  (background)

    Where:
        α = 2  reduces loss for easy positives (peak centres)
        β = 4  reduces loss for easy negatives (background far from peaks)

    Args:
        alpha:  Focal parameter for positive samples.
        beta:   Focal parameter for negative samples.
        reduction: 'mean' | 'sum'.
    """

    def __init__(self, alpha: float = 2.0, beta: float = 4.0,
                 pos_weight: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor,
                valid_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            pred:       [B, 1, H, W] predicted heatmap, values in [0, 1].
            target:     [B, 1, H, W] GT Gaussian heatmap, values in [0, 1].
            valid_mask: [B, 1, H, W] 1=supervised, 0=ignore.
                        If None, all pixels are supervised.

        Returns:
            scalar loss.
        """
        eps = 1e-6
        pred = pred.clamp(eps, 1 - eps)

        # Supervision mask (default: all pixels)
        if valid_mask is None:
            valid_mask = torch.ones_like(target)

        # Positive region: where target > 0 AND supervised
        pos_mask = ((target > 0.0) & (valid_mask > 0.5)).float()
        # Negative region: where target == 0 AND supervised
        neg_mask = ((target <= 0.0) & (valid_mask > 0.5)).float()

        # Focal weights (pos_weight amplifies positive samples)
        pos_loss = -pos_mask * ((1 - pred) ** self.alpha) * torch.log(pred)
        pos_loss = pos_loss * self.pos_weight
        neg_loss = -neg_mask * ((1 - target) ** self.beta) * \
                   (pred ** self.alpha) * torch.log(1 - pred)

        # Normalise by number of annotation peaks (target==1.0 at Gaussian centres),
        # NOT by the number of Gaussian pixels (which would dilute peak supervision).
        # Each pos_label annotation produces exactly one peak pixel with target=1.0.
        num_peaks = (target > 0.99).sum().clamp(min=1)
        loss = (pos_loss.sum() + neg_loss.sum()) / num_peaks
        return loss


# ======================================================================
#  Loss factory
# ======================================================================
def get_loss(config: dict) -> nn.Module:
    """Build a loss function from a YAML config dict.

    Supported ``loss`` values:
        - ``'cross_entropy'`` → ``CrossEntropyLoss(ignore_index=…)``
        - ``'focal'``         → ``FocalLoss(alpha=…, beta=…)``
    """
    loss_type = config.get('loss', 'cross_entropy')

    if loss_type == 'cross_entropy':
        ignore_idx = config.get('loss_ignore_index', 255)
        return nn.CrossEntropyLoss(ignore_index=ignore_idx)

    elif loss_type == 'focal':
        alpha = config.get('focal_alpha', 2.0)
        beta  = config.get('focal_beta', 4.0)
        pw    = config.get('focal_pos_weight', 1.0)
        return FocalLoss(alpha=alpha, beta=beta, pos_weight=pw)

    else:
        raise ValueError(f'Unknown loss type: {loss_type}. '
                         f'Choose "cross_entropy" or "focal".')
