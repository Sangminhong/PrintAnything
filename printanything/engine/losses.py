"""Training objective of PrintAnything (paper, Sec. 3.5).

    L   = lambda_M * L_M + lambda_R * L_R + lambda_Q * L_Q     Eq. (8)
    L_M = BCE(M', M) + DICE(M', M)                             Eq. (5)
    L_R = WCE(R', R)                                           Eq. (6)
    L_Q = Huber(Q', Q; M)                                      Eq. (7)
"""

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

__all__ = [
    "occupancy_loss",
    "region_loss",
    "flow_loss",
    "occupancy_accuracy",
    "binary_iou",
    "region_miou",
]


def occupancy_loss(logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """BCE + Dice on the occupancy map, Eq. (5).

    Args:
        logit: (B, 1, H, W) occupancy logits.
        target: (B, 1, H, W) binary ground truth.
    """
    target = target.float()
    bce = F.binary_cross_entropy_with_logits(logit, target)

    smooth = 1e-5
    p = torch.sigmoid(logit)
    inter = (p * target).sum()
    union = p.sum() + target.sum()
    dice = 1.0 - (2 * inter + smooth) / (union + smooth)
    return bce + dice


def region_loss(
    logits: torch.Tensor, target: torch.Tensor, class_weights: Optional[Sequence[float]] = None
) -> torch.Tensor:
    """Weighted cross-entropy on the region map, Eq. (6).

    Args:
        logits: (B, K, H, W) region logits.
        target: (B, H, W) integer class labels.
        class_weights: per-class weights; see ``REGION_CLASS_WEIGHTS``.
    """
    weight = (
        torch.tensor(class_weights, device=logits.device, dtype=torch.float32)
        if class_weights is not None
        else None
    )
    return F.cross_entropy(logits, target.long(), weight=weight)


def flow_loss(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, delta: float = 1.0
) -> torch.Tensor:
    """Huber loss on the flow map, restricted to printed pixels, Eq. (7)."""
    diff = pred - target
    absd = diff.abs()
    huber = torch.where(absd < delta, 0.5 * diff * diff, delta * (absd - 0.5 * delta))
    return (huber * mask).sum() / mask.sum().clamp_min(1.0)


@torch.no_grad()
def occupancy_accuracy(logit: torch.Tensor, target: torch.Tensor, thresh: float = 0.5) -> torch.Tensor:
    """Pixel accuracy of the thresholded occupancy prediction."""
    return ((torch.sigmoid(logit) > thresh) == (target > 0.5)).float().mean()


@torch.no_grad()
def binary_iou(logit: torch.Tensor, target: torch.Tensor, thresh: float = 0.0) -> float:
    """IoU of the occupancy map (``thresh`` applies to the logits)."""
    p = (logit > thresh).float()
    t = target.float()
    inter = (p * t).sum()
    union = (p + t - p * t).sum().clamp_min(1e-6)
    return float((inter / union).item())


@torch.no_grad()
def region_miou(
    logits: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None, num_classes: int = 5
) -> float:
    """Mean IoU over region classes; classes absent from both maps are skipped."""
    pred = logits.argmax(dim=1)
    ious = []
    for cls in range(num_classes):
        p = pred == cls
        t = target == cls
        if mask is not None:
            p, t = p & mask, t & mask
        union = (p | t).sum().item()
        if union == 0:
            continue
        ious.append((p & t).sum().item() / union)
    return float(np.mean(ious)) if ious else 0.0
