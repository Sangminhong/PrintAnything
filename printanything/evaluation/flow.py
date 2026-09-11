"""Extrusion-flow metrics (paper, Sec. 5.2, Eqs. 12-13).

These judge the *toolpaths* a G-plan map compiles to, not its geometry. Over the
extruding moves of the generated G-code (``dE_k > 0`` with non-zero XY motion),
with planar travel ``ds_k`` and extrusion density ``rho_k = dE_k / ds_k``:

    d(rho)-smooth = mean_k |rho_k - rho_{k-1}|                          Eq. (12)
    rho-CV        = mean over slices of  std(rho) / (|mean(rho)| + eps) Eq. (13)

Lower is better: both say the extruder deposits material at a steady rate rather
than surging along a path. They are what Table 4 uses to show that the flow map Q
matters even where the occupancy is already correct.
"""

import math
import re
from typing import Dict, Iterator, List, Tuple

import numpy as np

__all__ = ["parse_extrusion_moves", "gcode_flow_metrics", "gplan_flow_metrics"]

_EPS = 1e-9
_WORD = re.compile(r"([XYZEF])(-?\d*\.?\d+)", re.IGNORECASE)
_LAYER_TAG = re.compile(r";\s*LAYER:?\s*(\d+)", re.IGNORECASE)


def parse_extrusion_moves(gcode_path: str) -> Iterator[Tuple[int, float, float]]:
    """Yield ``(slice_index, ds_mm, dE_mm)`` for every extruding move.

    Handles both absolute (``M82``) and relative (``M83``) extrusion, and takes
    slice boundaries from ``;LAYER:`` markers when present, falling back to a Z
    change. Travel moves and pure retractions are skipped.
    """
    x = y = z = 0.0
    e_abs = 0.0
    relative_e = False
    slice_index = 0
    saw_layer_tag = False

    with open(gcode_path, "r", errors="ignore") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue

            if stripped.startswith(";"):
                tag = _LAYER_TAG.match(stripped)
                if tag:
                    saw_layer_tag = True
                    slice_index = int(tag.group(1))
                continue

            code = stripped.split()[0].upper()
            if code == "M83":
                relative_e = True
                continue
            if code == "M82":
                relative_e = False
                continue
            if code == "G92":
                for axis, value in _WORD.findall(stripped):
                    if axis.upper() == "E":
                        e_abs = float(value)
                continue
            if code not in ("G0", "G1"):
                continue

            words = {a.upper(): float(v) for a, v in _WORD.findall(stripped.split(";")[0])}
            new_x = words.get("X", x)
            new_y = words.get("Y", y)
            new_z = words.get("Z", z)

            if "E" not in words:
                dE = 0.0
            elif relative_e:
                dE = words["E"]
            else:
                dE = words["E"] - e_abs
                e_abs = words["E"]

            ds = math.hypot(new_x - x, new_y - y)
            if not saw_layer_tag and new_z > z + 1e-6:
                slice_index += 1

            x, y, z = new_x, new_y, new_z

            if dE > 0.0 and ds > _EPS:
                yield slice_index, ds, dE


def gcode_flow_metrics(gcode_path: str) -> Dict[str, float]:
    """``d(rho)-smooth`` and ``rho-CV`` of a generated G-code file.

    Returns:
        Dict with ``rho_smooth``, ``rho_cv``, ``num_moves`` and ``num_slices``.
        The two metrics are NaN when the file has too few extruding moves.
    """
    per_slice: Dict[int, List[float]] = {}
    rho_sequence: List[float] = []

    for slice_index, ds, dE in parse_extrusion_moves(gcode_path):
        rho = dE / (ds + _EPS)
        rho_sequence.append(rho)
        per_slice.setdefault(slice_index, []).append(rho)

    rho = np.asarray(rho_sequence, dtype=np.float64)
    smooth = float(np.mean(np.abs(np.diff(rho)))) if rho.size >= 2 else float("nan")

    cvs = [
        float(np.std(v) / (abs(np.mean(v)) + _EPS))
        for v in (np.asarray(values, dtype=np.float64) for values in per_slice.values())
        if v.size >= 1
    ]

    return {
        "rho_smooth": smooth,
        "rho_cv": float(np.mean(cvs)) if cvs else float("nan"),
        "num_moves": int(rho.size),
        "num_slices": len(per_slice),
    }


def gplan_flow_metrics(M, Q, px_mm: float) -> Dict[str, float]:
    """Flow smoothness read directly off a predicted G-plan map.

    A cheap stand-in for :func:`gcode_flow_metrics` that needs no compilation
    step: the flow map is sampled along the longest contour of each slice, and
    the same two statistics are computed on that sequence. Useful for monitoring
    during training, but the numbers are not comparable to the G-code ones.
    """
    from skimage import measure

    M = np.asarray(M.detach().cpu() if hasattr(M, "detach") else M).astype(np.uint8)
    Q = np.asarray(Q.detach().cpu() if hasattr(Q, "detach") else Q).astype(np.float64)
    if M.shape != Q.shape:
        return {"rho_smooth": float("nan"), "rho_cv": float("nan"), "num_slices": 0}

    px = max(float(px_mm), _EPS)
    deltas: List[np.ndarray] = []
    cvs: List[float] = []

    for z in range(M.shape[0]):
        layer = M[z]
        if layer.sum() < 2:
            continue
        contours = measure.find_contours(layer, level=0.5)
        if not contours:
            continue
        contour = max(contours, key=len)
        if len(contour) < 3:
            continue

        rows = np.clip(np.round(contour[:, 0]).astype(np.int64), 0, layer.shape[0] - 1)
        cols = np.clip(np.round(contour[:, 1]).astype(np.int64), 0, layer.shape[1] - 1)
        q_seq = Q[z, rows, cols]

        ds = np.hypot(np.diff(contour[:, 0]), np.diff(contour[:, 1])) * px
        valid = ds > _EPS
        if not np.any(valid):
            continue
        deltas.append(np.abs(np.diff(q_seq)[valid]))

        finite = q_seq[np.isfinite(q_seq)]
        if finite.size >= 2:
            cvs.append(float(np.std(finite) / (abs(np.mean(finite)) + _EPS)))

    return {
        "rho_smooth": float(np.mean(np.concatenate(deltas))) if deltas else float("nan"),
        "rho_cv": float(np.mean(cvs)) if cvs else float("nan"),
        "num_slices": len(cvs),
    }
