"""Scan-artifact augmentations (paper, Sec. 5.6).

Two augmentations make the model robust to the imperfections of commodity depth
sensors and 3D scanners:

``gaussian_jitter``
    Additive Gaussian noise in normalised coordinates (sigma = 0.01 in the paper).
``structured_hole_dropout``
    Removes every point inside a contiguous spherical region, mimicking the
    occlusion of a sensor that never observed part of the object.
"""

import torch

__all__ = ["gaussian_jitter", "structured_hole_dropout"]


def gaussian_jitter(pc: torch.Tensor, sigma: float) -> torch.Tensor:
    """Add zero-mean Gaussian noise to a (1, N, 3) point cloud in [-1, 1]."""
    if sigma <= 0.0:
        return pc
    return pc + torch.randn_like(pc) * float(sigma)


def structured_hole_dropout(
    pc: torch.Tensor,
    *,
    radius: float,
    target_frac: float,
    min_frac: float,
    max_frac: float,
    min_points_left: int = 1024,
    max_tries: int = 8,
) -> torch.Tensor:
    """Drop the points inside a spherical hole centred on a random point.

    The radius is adapted over at most ``max_tries`` attempts so that the removed
    fraction lands inside ``[min_frac, max_frac]`` and as close as possible to
    ``target_frac``.

    Args:
        pc: (1, N, 3) point cloud in normalised coordinates.

    Returns:
        (1, N', 3) with N' <= N; the input is returned unchanged when no
        acceptable hole was found or too few points would remain.
    """
    assert pc.dim() == 3 and pc.shape[0] == 1 and pc.shape[2] == 3, "expected (1, N, 3)"
    N = int(pc.shape[1])
    if N <= min_points_left:
        return pc

    target_frac = float(max(0.0, min(0.9, target_frac)))
    min_frac = float(max(0.0, min(0.9, min_frac)))
    max_frac = float(max(0.0, min(0.9, max_frac)))
    max_frac = max(max_frac, min_frac)

    pts = pc[0]
    r = float(max(1e-6, radius))
    best_keep, best_dist = None, float("inf")

    for _ in range(int(max_tries)):
        center = pts[torch.randint(low=0, high=N, size=(1,), device=pc.device).item()]
        d2 = ((pts - center[None, :]) ** 2).sum(dim=1)
        keep = d2 > (r * r)
        removed = float((~keep).float().mean().item())

        if removed > max_frac:
            r *= 0.8
        elif removed < min_frac:
            r *= 1.25

        dist = abs(removed - target_frac)
        if dist < best_dist and int(keep.sum().item()) >= min_points_left:
            best_dist, best_keep = dist, keep
        if (min_frac <= removed <= max_frac) and dist < 0.02:
            best_keep = keep
            break

    if best_keep is None:
        return pc
    kept = pts[best_keep]
    if int(kept.shape[0]) < min_points_left:
        return pc
    return kept.unsqueeze(0)
