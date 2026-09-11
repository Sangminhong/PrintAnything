"""Infill pattern dictionary and template insertion (paper, Sec. 3.3).

Rather than learning infill geometry from scratch, PrintAnything keeps a small
library of hand-designed templates (grid, cubic, honeycomb, gyroid) that ship in
``assets/infill_patterns``. A template is warped to the slice resolution with
periodic wrapping -- so it tiles seamlessly at any scale, rotation or offset --
and then inserted **only** inside the infill-designated pixels of the predicted
region map. Wall, support and skirt pixels are left untouched, which is what
keeps the structural shell of the object intact.
"""

import math
import os
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..gplan.constants import INFILL, STRUCTURAL_CLASSES

__all__ = [
    "PATTERNS",
    "MONOTONIC",
    "default_pattern_dir",
    "load_template",
    "warp_template_periodic",
    "apply_infill",
]

#: Solid fill, represented by an all-ones template (top/bottom shells).
MONOTONIC = "monotonic"

#: Pattern dictionary. ``monotonic`` is synthesised, the rest are ``.npy`` files.
PATTERNS: Tuple[str, ...] = (MONOTONIC, "cubic", "grid", "honeycomb", "gyroid")


def default_pattern_dir() -> str:
    """``assets/infill_patterns`` next to the repository root."""
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "assets",
        "infill_patterns",
    )


def load_template(
    pattern: str, patterns_dir: Optional[str] = None, device=None
) -> torch.Tensor:
    """Load one binary pattern template as a (1, 1, h, w) float tensor in {0, 1}."""
    device = device or torch.device("cpu")
    if pattern == MONOTONIC:
        return torch.ones((1, 1, 32, 32), device=device, dtype=torch.float32)

    patterns_dir = patterns_dir or default_pattern_dir()
    path = os.path.join(patterns_dir, f"{pattern}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Infill template '{pattern}' not found at {path}. "
            f"Available patterns: {', '.join(PATTERNS)}"
        )
    arr = (np.load(path) > 0).astype(np.float32)
    return torch.from_numpy(arr)[None, None, ...].to(device=device, dtype=torch.float32)


def warp_template_periodic(
    template: torch.Tensor,
    *,
    H: int,
    W: int,
    scale: float,
    theta_deg: float = 0.0,
    dx_px: float = 0.0,
    dy_px: float = 0.0,
) -> torch.Tensor:
    """Scale/rotate/translate a template over an (H, W) grid with periodic wrap.

    The sampling grid is wrapped into [-1, 1] before ``grid_sample``, so the
    pattern tiles seamlessly instead of leaving empty borders. Smaller ``scale``
    means a finer pattern (more repetitions per slice).
    """
    tiled = F.interpolate(template, size=(H, W), mode="bilinear", align_corners=False).clamp(0.0, 1.0)
    dtype, device = tiled.dtype, tiled.device

    gy, gx = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    gy = gy - (H - 1) * 0.5
    gx = gx - (W - 1) * 0.5

    theta = torch.tensor(theta_deg * math.pi / 180.0, device=device, dtype=dtype)
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    s = torch.tensor(float(scale), device=device, dtype=dtype)

    u = s * (cos_t * gx - sin_t * gy) + ((W - 1) * 0.5 + float(dx_px))
    v = s * (sin_t * gx + cos_t * gy) + ((H - 1) * 0.5 + float(dy_px))

    gu = torch.remainder((u / max(1.0, W - 1)) * 2.0 - 1.0 + 1.0, 2.0) - 1.0
    gv = torch.remainder((v / max(1.0, H - 1)) * 2.0 - 1.0 + 1.0, 2.0) - 1.0

    grid = torch.stack([gu, gv], dim=-1)[None, ...]
    warped = F.grid_sample(tiled, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return warped.clamp(0.0, 1.0)


def apply_infill(
    M: np.ndarray,
    R: np.ndarray,
    *,
    pattern: str,
    scale: float,
    theta_deg: float = 0.0,
    dx_px: float = 0.0,
    dy_px: float = 0.0,
    threshold: float = 0.5,
    patterns_dir: Optional[str] = None,
    device=None,
    structural_classes: Sequence[int] = STRUCTURAL_CLASSES,
) -> Tuple[np.ndarray, np.ndarray]:
    """Synthesise the infill of a predicted G-plan map.

    Args:
        M: (Z, H, W) predicted occupancy.
        R: (Z, H, W) predicted region map.
        pattern, scale, theta_deg, dx_px, dy_px: the infill policy to realise.
        threshold: binarisation threshold of the warped template.

    Returns:
        ``(M_infilled, R_infilled)``, both (Z, H, W) uint8. Structural pixels keep
        their class; every other pixel becomes infill exactly where the warped
        template fires.
    """
    device = device or torch.device("cpu")
    Z, H, W = R.shape
    template = load_template(pattern, patterns_dir, device=device)
    structural = np.asarray(structural_classes, dtype=np.uint8)

    # The warp does not depend on the slice, so compute it once.
    warped = warp_template_periodic(
        template, H=H, W=W, scale=float(scale), theta_deg=theta_deg, dx_px=dx_px, dy_px=dy_px
    )[0, 0].cpu().numpy()

    M_out = np.zeros_like(M, dtype=np.uint8)
    R_out = np.zeros_like(R, dtype=np.uint8)

    for zi in range(Z):
        R_slice = R[zi].astype(np.uint8)
        is_structural = np.isin(R_slice, structural)
        infill_area = (R_slice == INFILL).astype(np.float32)

        fired = (warped * infill_area) > float(threshold)

        R_new = R_slice.copy()
        R_new[~is_structural] = fired[~is_structural].astype(np.uint8) * INFILL
        R_out[zi] = R_new
        M_out[zi] = (is_structural | fired).astype(np.uint8)

    return M_out, R_out
