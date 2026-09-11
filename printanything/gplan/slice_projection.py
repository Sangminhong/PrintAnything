"""Slice-wise point projection and Multi-Slice Conditioning (paper, Sec. 3.2).

An unstructured point cloud is turned into a slice-aligned 2D input by selecting
the points that fall inside the slab of the target slice, together with the
``z_context`` slabs above and below it (Multi-Slice Conditioning, MSC), and
rasterising each slab onto the printer XY grid.

Every slab contributes three channels

    occ       {0, 1}     whether any point falls into the pixel
    density   [0, 1]     log-normalised point count per pixel
    mean_zoff [-1, 1]    mean signed distance to the slab centre, in slab units

so the projected tensor has ``(2 * z_context + 1) * 3`` channels. ``z_context=0``
disables MSC and reproduces the ablation row of Table 2.

Two projections are provided:

``project_slice``
    Rasterises directly in the normalised ``[-1, 1]`` cube. Use it whenever the
    input is a bare point cloud (inference on user data).
``project_slice_printer_frame``
    Rasterises in the printer frame of the ground-truth G-plan map so that the
    projection is pixel-aligned with the supervision. Use it during training and
    evaluation on Slice-100K, where the dataset provides the raster metadata.
"""

from typing import List, Sequence, Tuple

import torch

__all__ = [
    "num_projection_channels",
    "project_slice",
    "project_slice_printer_frame",
]

_EPS = 1e-6


def num_projection_channels(z_context: int = 1) -> int:
    """Channel count of the slice-aligned input for a given MSC context."""
    return (2 * int(z_context) + 1) * 3


def _slab_channels(
    lin_idx_all: torch.Tensor,
    z_all: torch.Tensor,
    z_centers: Sequence[torch.Tensor],
    dz: torch.Tensor,
    H: int,
    W: int,
    device,
) -> torch.Tensor:
    """Scatter the points of each slab into (occ, density, mean_zoff) planes."""
    channels: List[torch.Tensor] = []
    for z_center in z_centers:
        lower = z_center - 0.5 * dz
        upper = z_center + 0.5 * dz
        slab = (z_all >= lower) & (z_all < upper)

        if int(slab.sum().item()) == 0:
            channels.extend(torch.zeros(1, H, W, device=device) for _ in range(3))
            continue

        lin_idx = lin_idx_all[slab]
        z_slab = z_all[slab]

        count = torch.zeros(H * W, device=device)
        count.index_add_(0, lin_idx, torch.ones(lin_idx.numel(), device=device, dtype=count.dtype))

        zoff = (z_slab - z_center) / (dz + _EPS)  # approximately [-0.5, 0.5]
        sum_zoff = torch.zeros(H * W, device=device)
        sum_zoff.index_add_(0, lin_idx, zoff.to(sum_zoff.dtype))

        occ = (count.view(H, W) > 0).float()
        density = torch.log1p(count).view(H, W)
        density = density / density.max().clamp_min(_EPS)
        mean_zoff = (sum_zoff / count.clamp_min(1.0)).view(H, W).clamp(-1.0, 1.0)

        channels.extend([occ.unsqueeze(0), density.unsqueeze(0), mean_zoff.unsqueeze(0)])

    return torch.stack(channels, dim=1)  # (1, C, H, W)


def project_slice(
    pc_norm: torch.Tensor,
    z_center_norm01: float,
    dz_norm01: float,
    H: int,
    W: int,
    device,
    z_context: int = 1,
) -> torch.Tensor:
    """Project a normalised point cloud onto the XY grid of one slice.

    Args:
        pc_norm: (N, 3) points normalised to [-1, 1].
        z_center_norm01: slice centre along z, in [0, 1].
        dz_norm01: slab thickness as a fraction of the object height.
        H, W: raster resolution.
        z_context: number of neighbouring slabs on each side (MSC).

    Returns:
        (1, C, H, W) float tensor with ``C = num_projection_channels(z_context)``.
    """
    C = num_projection_channels(z_context)
    if pc_norm.numel() == 0:
        return torch.zeros(1, C, H, W, device=device)

    # [0, 1] slice coordinate -> [-1, 1] point-cloud coordinate.
    base_center = torch.as_tensor(
        z_center_norm01 * 2.0 - 1.0, device=device, dtype=pc_norm.dtype
    )
    dz = torch.as_tensor(dz_norm01 * 2.0, device=device, dtype=pc_norm.dtype).clamp_min(_EPS)

    xs = ((pc_norm[:, 0] + 1.0) * 0.5 * W).long().clamp_(0, W - 1)
    ys = ((pc_norm[:, 1] + 1.0) * 0.5 * H).long().clamp_(0, H - 1)

    z_centers = [base_center + float(k) * dz for k in range(-z_context, z_context + 1)]
    return _slab_channels(ys * W + xs, pc_norm[:, 2], z_centers, dz, H, W, device)


def project_slice_printer_frame(
    pc_norm: torch.Tensor,
    ctr_mm,
    scale_mm: float,
    z_center_norm01: float,
    dz_norm01: float,
    out_H: int,
    out_W: int,
    device,
    *,
    z_context: int = 1,
    origin_xy_mm: Tuple[float, float] = (0.0, 0.0),
    px_mm: float = 0.2,
    raster_H: int = 256,
    raster_W: int = 256,
    scale_to_target: float = 1.0,
    pad_oy: int = 0,
    pad_ox: int = 0,
) -> torch.Tensor:
    """Project onto the XY grid of the ground-truth G-plan raster.

    The point cloud is de-normalised back to millimetres, mapped to raw G-plan
    pixels with ``x_px = (x_mm - origin_x) / px_mm``, and then pushed through the
    same downsample-to-fit plus centre-pad transform that the dataset applies to
    the supervision, so the projection is pixel-aligned with ``M``/``R``/``Q``.

    The slicer recentres the part on the print bed, so the bounding-box centre of
    the point cloud is matched to the centre of the G-plan raster before the
    mapping; without this the projection collapses onto the raster border.
    """
    C = num_projection_channels(z_context)
    if pc_norm.numel() == 0:
        return torch.zeros(1, C, out_H, out_W, device=device)

    if not torch.is_tensor(ctr_mm):
        ctr_mm = torch.tensor(ctr_mm, device=device, dtype=pc_norm.dtype)
    else:
        ctr_mm = ctr_mm.to(device=device, dtype=pc_norm.dtype)

    pc_norm = pc_norm.to(device=device)
    scale_t = torch.tensor(float(scale_mm), device=device, dtype=pc_norm.dtype)
    pc_mm = pc_norm * scale_t + ctr_mm[None, :]
    x_mm, y_mm = pc_mm[:, 0], pc_mm[:, 1]
    z_all = pc_norm[:, 2]  # slab selection stays in normalised z

    origin_x, origin_y = float(origin_xy_mm[0]), float(origin_xy_mm[1])
    px = float(px_mm)

    # Match bounding-box centres (point cloud frame -> printer frame).
    ir_cx_mm = origin_x + 0.5 * float(raster_W) * px
    ir_cy_mm = origin_y + 0.5 * float(raster_H) * px
    x_mm = x_mm + (ir_cx_mm - 0.5 * (x_mm.min() + x_mm.max()))
    y_mm = y_mm + (ir_cy_mm - 0.5 * (y_mm.min() + y_mm.max()))

    # Raw G-plan pixel indices; points outside the raster are dropped rather than
    # clamped onto the border.
    xs_raw_f = torch.floor((x_mm - origin_x) / (px + _EPS))
    ys_raw_f = torch.floor((y_mm - origin_y) / (px + _EPS))
    inside = (
        (xs_raw_f >= 0)
        & (xs_raw_f < float(raster_W))
        & (ys_raw_f >= 0)
        & (ys_raw_f < float(raster_H))
    )
    xs_raw = xs_raw_f.long()[inside]
    ys_raw = ys_raw_f.long()[inside]
    z_in = z_all[inside]

    # Slab geometry follows the actual z-extent of the point cloud: thin objects
    # do not span the full [-1, 1] range and would otherwise yield empty slabs.
    if z_in.numel() > 0:
        z_min, z_max = z_in.min(), z_in.max()
        z_span = (z_max - z_min).clamp_min(_EPS)
        base_center = z_min + z_span * float(z_center_norm01)
        dz = (z_span * float(dz_norm01)).clamp_min(_EPS)
    else:
        base_center = torch.as_tensor(
            float(z_center_norm01) * 2.0 - 1.0, device=device, dtype=z_all.dtype
        )
        dz = torch.as_tensor(
            float(dz_norm01) * 2.0, device=device, dtype=z_all.dtype
        ).clamp_min(_EPS)

    # Same downsample-to-fit + centre-pad mapping as the dataset loader.
    s = float(scale_to_target)
    if s < 1.0:
        W_res = max(1, int(round(int(raster_W) * s)))
        H_res = max(1, int(round(int(raster_H) * s)))
        xs = torch.floor(xs_raw.to(pc_norm.dtype) * s).long().clamp_(0, W_res - 1)
        ys = torch.floor(ys_raw.to(pc_norm.dtype) * s).long().clamp_(0, H_res - 1)
    else:
        xs, ys = xs_raw, ys_raw

    xs = (xs + int(pad_ox)).clamp_(0, out_W - 1)
    ys = (ys + int(pad_oy)).clamp_(0, out_H - 1)

    z_centers = [base_center + float(k) * dz for k in range(-z_context, z_context + 1)]
    return _slab_channels(ys * out_W + xs, z_in, z_centers, dz, out_H, out_W, device)
