"""Strength and cost proxies of an infilled G-plan map (paper, Sec. 3.3, 5.5).

Evaluating an infill policy by actually printing it is not an option, so two
cheap proxies are computed directly from the synthesised maps.

Strength ``S`` in [0, 1] rewards
    * the infill material ratio inside the occupied region,
    * the connectivity of the infill (largest connected component / total),
    * direction balance, i.e. infill boundaries spread over both axes rather
      than running only one way.

Cost ``C`` grows with the printed area and with the number of infill boundary
transitions, which stands in for toolpath complexity and printing overhead:

    material cost = occupied pixels
    time cost     = occupied pixels + 0.5 * boundary transitions

The combined score reported in Table 3 is ``1e5 * S / C``.
"""

from collections import deque
from typing import Dict

import numpy as np

from ..gplan.constants import INFILL, SKIRT, SUPPORT, WALL

__all__ = [
    "STRENGTH_WEIGHTS",
    "largest_component_ratio",
    "compute_proxies",
    "combined_score",
]

#: Weights of the three strength terms (infill ratio, connectivity, direction).
STRENGTH_WEIGHTS = (0.5, 0.3, 0.2)


def largest_component_ratio(binary: np.ndarray) -> float:
    """Fraction of the set pixels that lie in its largest 4-connected component."""
    total = int(binary.sum())
    if total <= 0:
        return 0.0

    h, w = binary.shape
    visited = np.zeros_like(binary, dtype=bool)
    best = 0
    queue: deque = deque()

    for y0 in range(h):
        for x0 in range(w):
            if binary[y0, x0] == 0 or visited[y0, x0]:
                continue
            visited[y0, x0] = True
            queue.append((y0, x0))
            size = 0
            while queue:
                cy, cx = queue.popleft()
                size += 1
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and binary[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))
            best = max(best, size)

    return float(best) / float(total)


def compute_proxies(M: np.ndarray, R: np.ndarray) -> Dict[str, float]:
    """Proxy objectives of an infilled G-plan map.

    Args:
        M: (Z, H, W) occupancy.
        R: (Z, H, W) region map.

    Returns:
        Dict with ``strength_proxy``, ``cost_material``, ``cost_time`` and the
        diagnostics they are built from.
    """
    occ_total = infill_total = structural_total = transitions_total = 0
    connectivity, direction = [], []

    for zi in range(R.shape[0]):
        M_slice = M[zi]
        R_slice = R[zi]

        infill_bin = (R_slice == INFILL).astype(np.uint8)
        occ_total += int((M_slice > 0).sum())
        infill_total += int(infill_bin.sum())
        structural_total += int(np.isin(R_slice, (WALL, SUPPORT, SKIRT)).sum())

        t_h = int(np.abs(np.diff(infill_bin, axis=1)).sum()) if infill_bin.shape[1] > 1 else 0
        t_v = int(np.abs(np.diff(infill_bin, axis=0)).sum()) if infill_bin.shape[0] > 1 else 0
        transitions_total += t_h + t_v

        connectivity.append(largest_component_ratio(infill_bin))
        direction.append(float(min(t_h, t_v) / max(max(t_h, t_v), 1)))

    occ_safe = max(occ_total, 1)
    infill_ratio = infill_total / occ_safe
    connectivity_score = float(np.mean(connectivity)) if connectivity else 0.0
    direction_balance = float(np.mean(direction)) if direction else 0.0

    w_r, w_c, w_d = STRENGTH_WEIGHTS
    strength = (
        w_r * min(max(infill_ratio, 0.0), 1.0)
        + w_c * min(max(connectivity_score, 0.0), 1.0)
        + w_d * min(max(direction_balance, 0.0), 1.0)
    )

    return {
        "strength_proxy": float(strength),
        "cost_material": float(occ_total),
        "cost_time": float(occ_total + 0.5 * transitions_total),
        "infill_ratio": float(infill_ratio),
        "connectivity_score": connectivity_score,
        "direction_balance": direction_balance,
        "occupied_pixels": float(occ_total),
        "infill_pixels": float(infill_total),
        "structural_pixels": float(structural_total),
        "transition_count": float(transitions_total),
    }


def combined_score(strength: float, cost: float) -> float:
    """``1e5 * S / C``: the strength-per-cost figure reported in Table 3."""
    return 1e5 * float(strength) / max(float(cost), 1e-9)
