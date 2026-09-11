"""Training and validation loops for GPNet.

One optimisation step consumes a single object: the point cloud is encoded once
and a random subset of its slices is supervised (``slices_per_step``), which
keeps memory bounded for tall objects while still covering the whole height
range over the course of an epoch. The first and last slice are always included
because they carry the solid bottom/top shells.
"""

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from tqdm import tqdm

from ..datasets.augment import gaussian_jitter, structured_hole_dropout
from ..gplan.constants import NUM_REGION_CLASSES, REGION_CLASS_WEIGHTS
from .losses import binary_iou, flow_loss, occupancy_loss, region_loss, region_miou

__all__ = ["AugmentConfig", "LossConfig", "collate_objects", "raster_meta_from_batch",
           "train_one_epoch", "validate"]


def _grad_scaler(device, enabled: bool):
    """GradScaler across torch versions (``torch.amp`` since 2.3, ``torch.cuda.amp`` before)."""
    enabled = bool(enabled) and device.type == "cuda"
    try:
        return torch.amp.GradScaler(device.type, enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def collate_objects(batch):
    """Objects differ in slice count and raster size, so they stay unbatched."""
    return batch


def raster_meta_from_batch(sample: Dict) -> Optional[Dict]:
    """Extract the printer-frame raster metadata needed to align the projection.

    Returns None when the sample carries no raster metadata, in which case the
    slice projection falls back to the normalised cube.
    """
    if "raster_origin_xy_mm" not in sample or "raster_px_mm" not in sample:
        return None
    H, W = sample["M"].shape[1], sample["M"].shape[2]
    return {
        "ctr_mm": sample.get("ctr_mm", torch.zeros(3)),
        "scale_mm": sample.get("scale_mm", 1.0),
        "origin_xy_mm": sample["raster_origin_xy_mm"],
        "px_mm": sample["raster_px_mm"],
        "raster_H": sample.get("raster_H", H),
        "raster_W": sample.get("raster_W", W),
        "scale_to_target": sample.get("raster_scale_to_target", 1.0),
        "pad_oy": sample.get("raster_pad_oy", 0),
        "pad_ox": sample.get("raster_pad_ox", 0),
    }


@dataclass
class AugmentConfig:
    """Scan-artifact augmentation schedule (paper, Sec. 5.6).

    ``*_every`` counts optimisation steps: the augmentation fires whenever
    ``step % every == 0``; 0 disables it. When a list of magnitudes is given, one
    is drawn uniformly per firing so that a single model covers several
    corruption levels.
    """

    noise_every: int = 0
    noise_sigma: float = 0.0
    noise_sigmas: Optional[Sequence[float]] = None
    hole_every: int = 0
    hole_radius: float = 0.12
    hole_target_frac: float = 0.06
    hole_min_frac: float = 0.02
    hole_max_frac: float = 0.12
    hole_fracs: Optional[Sequence[float]] = None

    def __call__(self, pc: torch.Tensor, step: int) -> torch.Tensor:
        if self.noise_every and step % int(self.noise_every) == 0:
            sigma = random.choice(self.noise_sigmas) if self.noise_sigmas else float(self.noise_sigma)
            if sigma > 0:
                pc = gaussian_jitter(pc, sigma).clamp(-1.0, 1.0)

        if self.hole_every and step % int(self.hole_every) == 0:
            frac = random.choice(self.hole_fracs) if self.hole_fracs else float(self.hole_target_frac)
            if frac > 0:
                if self.hole_fracs:
                    lo, hi = max(0.0, frac - 0.02), min(0.9, frac + 0.02)
                else:
                    lo, hi = float(self.hole_min_frac), float(self.hole_max_frac)
                pc = structured_hole_dropout(
                    pc, radius=float(self.hole_radius), target_frac=float(frac),
                    min_frac=lo, max_frac=hi,
                )
        return pc


@dataclass
class LossConfig:
    """Weights of Eq. (8) and the choice of flow objective.

    ``flow_loss='huber'`` is the masked Huber loss of Eq. (7). ``'l1_sum'`` is the
    unnormalised L1 used by the original training runs; it needs a much smaller
    ``lambda_Q`` (3e-5). See ``docs/REPRODUCE.md``.
    """

    lambda_M: float = 1.0
    lambda_R: float = 1.0
    lambda_Q: float = 1.0
    flow_loss: str = "huber"
    region_class_weights: Sequence[float] = field(default_factory=lambda: list(REGION_CLASS_WEIGHTS))

    def flow(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.flow_loss == "huber":
            return flow_loss(pred, target, mask)
        if self.flow_loss == "l1_sum":
            return (pred - target).abs().sum()
        raise ValueError(f"Unknown flow loss '{self.flow_loss}' (use 'huber' or 'l1_sum')")


def _select_slice_indices(num_slices: int, budget: int) -> List[int]:
    """First and last slice plus a random middle subset, sorted and unique."""
    if num_slices < 2:
        return [0]
    middle = list(range(1, num_slices - 1))
    k = min(max(0, budget - 2), len(middle))
    return sorted({0, num_slices - 1, *(random.sample(middle, k) if k > 0 else [])})


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    *,
    loss_cfg: Optional[LossConfig] = None,
    augment: Optional[AugmentConfig] = None,
    slices_per_step: int = 22,
    slice_batch_size: int = 4,
    amp: bool = True,
    use_prev_context: bool = False,
    skip_oom: bool = True,
) -> Dict[str, float]:
    """Run one epoch and return the mean loss terms and occupancy IoU."""
    loss_cfg = loss_cfg or LossConfig()
    model.train()
    amp = bool(amp) and device.type == "cuda"
    scaler = _grad_scaler(device, amp)

    running = {"loss": 0.0, "loss_M": 0.0, "loss_R": 0.0, "loss_Q": 0.0, "iou_M": 0.0, "n": 0}
    pbar = tqdm(loader, desc="train", ncols=100)

    for step, batch in enumerate(pbar):
        sample = batch[0]
        try:
            pc = sample["pc"].to(device).unsqueeze(0)  # (1, N, 3)
            if augment is not None:
                pc = augment(pc, step)

            M_gt = sample["M"].to(device)  # (Z, H, W)
            R_gt = sample["R"].to(device) if "R" in sample else None
            Q_gt = sample["Q"].to(device) if "Q" in sample else None

            Z, H, W = M_gt.shape
            indices = _select_slice_indices(Z, slices_per_step)
            n_sel = len(indices)

            z_values = torch.tensor([i / (Z - 1) if Z > 1 else 0.0 for i in indices])
            target_shapes = [(H, W)] * n_sel

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=amp):
                M_pred, R_pred, Q_pred = model(
                    pc,
                    z_values,
                    target_shapes,
                    slice_batch_size=min(slice_batch_size, n_sel),
                    use_prev_context=use_prev_context,
                    raster_meta=raster_meta_from_batch(sample),
                )

                loss = torch.zeros((), device=device)
                acc = {k: torch.zeros((), device=device) for k in ("M", "R", "Q", "iou")}

                for i, z_idx in enumerate(indices):
                    M_target = M_gt[z_idx].unsqueeze(0).unsqueeze(0)

                    l_M = occupancy_loss(M_pred[i], M_target)
                    acc["M"] += l_M
                    loss = loss + loss_cfg.lambda_M * l_M

                    pred_bin = (M_pred[i] > 0).float()
                    tgt = M_target.float()
                    inter = (pred_bin * tgt).sum()
                    acc["iou"] += inter / (pred_bin + tgt - pred_bin * tgt).sum().clamp_min(1e-6)

                    if R_pred is not None and R_gt is not None:
                        l_R = region_loss(
                            R_pred[i], R_gt[z_idx].unsqueeze(0), loss_cfg.region_class_weights
                        )
                        acc["R"] += l_R
                        loss = loss + loss_cfg.lambda_R * l_R

                    if Q_pred is not None and Q_gt is not None:
                        l_Q = loss_cfg.flow(
                            Q_pred[i], Q_gt[z_idx].unsqueeze(0).unsqueeze(0), (M_target > 0).float()
                        )
                        acc["Q"] += l_Q
                        loss = loss + loss_cfg.lambda_Q * l_Q

                loss = loss / n_sel
                for k in acc:
                    acc[k] = acc[k] / n_sel

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if device.type == "cuda":
                torch.cuda.empty_cache()

            running["loss"] += float(loss.item())
            running["loss_M"] += float(acc["M"].item())
            running["loss_R"] += float(acc["R"].item())
            running["loss_Q"] += float(acc["Q"].item())
            running["iou_M"] += float(acc["iou"].item())
            running["n"] += 1
            pbar.set_postfix(
                loss=running["loss"] / running["n"], iou_M=running["iou_M"] / running["n"]
            )

        except RuntimeError as exc:  # pragma: no cover - depends on GPU state
            if skip_oom and "out of memory" in str(exc).lower():
                print(f"\n[train] OOM on sample {step}; skipping.")
                torch.cuda.empty_cache()
                continue
            raise

    n = max(1, running["n"])
    return {k: (v / n if k != "n" else v) for k, v in running.items()}


@torch.no_grad()
def validate(model, loader, device, *, slice_batch_size: int = 8, use_prev_context: bool = False) -> Dict[str, float]:
    """Slice-wise IoU(M), mIoU(R) and masked L1(Q) over the whole loader."""
    model.eval()
    ious, mious, l1_flow = [], [], []

    for batch in loader:
        sample = batch[0]
        pc = sample["pc"].to(device).unsqueeze(0)
        M_gt = sample["M"].to(device)
        R_gt = sample["R"].to(device) if "R" in sample else None
        Q_gt = sample["Q"].to(device) if "Q" in sample else None

        Z, H, W = M_gt.shape
        M_pred, R_pred, Q_pred = model(
            pc,
            torch.linspace(0, 1, Z),
            [(H, W)] * Z,
            slice_batch_size=slice_batch_size,
            use_prev_context=use_prev_context,
            raster_meta=raster_meta_from_batch(sample),
        )

        for i in range(Z):
            M_target = M_gt[i].unsqueeze(0).unsqueeze(0)
            ious.append(binary_iou(M_pred[i], M_target, 0.0))

            if R_pred is not None and R_gt is not None:
                mious.append(
                    region_miou(
                        R_pred[i],
                        R_gt[i].unsqueeze(0),
                        mask=(M_target[:, 0] > 0),
                        num_classes=NUM_REGION_CLASSES,
                    )
                )

            if Q_pred is not None and Q_gt is not None:
                mask = (M_target > 0).float()
                Q_target = Q_gt[i].unsqueeze(0).unsqueeze(0)
                l1_flow.append(
                    float((((Q_pred[i] - Q_target).abs() * mask).sum() / mask.sum().clamp_min(1.0)).item())
                )

    return {
        "IoU_M": float(np.mean(ious)) if ious else 0.0,
        "mIoU_R": float(np.mean(mious)) if mious else 0.0,
        "L1_Q": float(np.mean(l1_flow)) if l1_flow else 0.0,
    }
