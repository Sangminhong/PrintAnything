"""End-to-end inference: point cloud -> G-plan map -> G-code.

``predict_gplan`` runs GPNet over every slice of one object and returns the three
G-plan maps as numpy arrays. ``point_cloud_to_gcode`` chains the whole pipeline of
Figure 2 -- prediction, infill synthesis, and slice-wise compilation -- which is
what ``tools/infer.py`` exposes on the command line.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from .gplan.constants import RASTER_SIZE
from .gplan.gplan_to_gcode import GCodeConfig, gplan_to_gcode
from .gplan.io import save_gplan_folder
from .infill.patterns import apply_infill

__all__ = ["GPlanPrediction", "predict_gplan", "predict_from_points", "point_cloud_to_gcode"]


@dataclass
class GPlanPrediction:
    """The predicted G-plan map of one object."""

    M: np.ndarray  # (Z, H, W) uint8 occupancy
    R: Optional[np.ndarray]  # (Z, H, W) uint8 region
    Q: Optional[np.ndarray]  # (Z, H, W) float32 flow, in [0, 1]
    px_mm: float = 0.2
    dz_mm: float = 0.2

    def __len__(self) -> int:
        return int(self.M.shape[0])


def _to_numpy(x) -> np.ndarray:
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


@torch.no_grad()
def predict_gplan(
    model,
    pc: torch.Tensor,
    num_slices: int,
    *,
    device,
    raster_size: Tuple[int, int] = RASTER_SIZE,
    slice_batch_size: int = 8,
    raster_meta: Optional[Dict] = None,
    px_mm: float = 0.2,
    dz_mm: float = 0.2,
    flow_range: Optional[Tuple[float, float]] = (0.0, 1.0),
) -> GPlanPrediction:
    """Predict the G-plan map of one object.

    Args:
        model: a trained :class:`~printanything.models.GPNet`, in eval mode.
        pc: (N, 3) or (1, N, 3) point cloud normalised to [-1, 1].
        num_slices: number of slices to predict.
        raster_size: (H, W) of each predicted map.
        raster_meta: printer-frame metadata; see
            :func:`printanything.engine.raster_meta_from_batch`.
        px_mm, dz_mm: physical size of a pixel and of a slice, recorded on the
            prediction so that the compiler can place the toolpaths.
        flow_range: the flow head is an unbounded regressor, so its output is
            clamped into the range the ground-truth flow map lives in; None
            leaves the raw values alone. Without this, a slice predicted below
            zero would compile to toolpaths that extrude nothing.
    """
    model.eval()
    if pc.dim() == 2:
        pc = pc.unsqueeze(0)
    pc = pc.to(device)

    M_list, R_list, Q_list = model(
        pc,
        torch.linspace(0, 1, num_slices),
        [tuple(raster_size)] * num_slices,
        slice_batch_size=slice_batch_size,
        use_prev_context=False,
        raster_meta=raster_meta,
    )

    M = (np.stack([_to_numpy(m[0, 0]) for m in M_list], axis=0) > 0).astype(np.uint8)
    R = (
        np.stack([_to_numpy(r[0].argmax(0)) for r in R_list], axis=0).astype(np.uint8)
        if R_list
        else None
    )
    Q = (
        np.stack([_to_numpy(q[0, 0]) for q in Q_list], axis=0).astype(np.float32)
        if Q_list
        else None
    )
    if Q is not None and flow_range is not None:
        Q = np.clip(Q, flow_range[0], flow_range[1])
    return GPlanPrediction(M=M, R=R, Q=Q, px_mm=float(px_mm), dz_mm=float(dz_mm))


def predict_from_points(
    model,
    points: np.ndarray,
    *,
    device,
    num_slices: Optional[int] = None,
    layer_height_mm: float = 0.2,
    bed_px_mm: float = 0.2,
    raster_size: Tuple[int, int] = RASTER_SIZE,
    slice_batch_size: int = 8,
) -> Tuple[GPlanPrediction, Dict]:
    """Predict from a raw, unnormalised point cloud in millimetres.

    The cloud is centred and scaled to [-1, 1] the way the dataset does it, and
    the slice count follows the physical height of the object at
    ``layer_height_mm``. Returns the prediction and the normalisation it used.
    """
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if points.size == 0:
        raise ValueError("Empty point cloud.")

    center = 0.5 * (points.min(axis=0) + points.max(axis=0))
    extent = float(np.abs(points - center).max())
    scale = extent if extent > 1e-9 else 1.0
    pc_norm = (points - center) / scale

    height_mm = float(points[:, 2].max() - points[:, 2].min())
    if num_slices is None:
        num_slices = max(1, int(round(height_mm / max(layer_height_mm, 1e-6))))

    prediction = predict_gplan(
        model,
        torch.from_numpy(pc_norm),
        num_slices,
        device=device,
        raster_size=raster_size,
        slice_batch_size=slice_batch_size,
        px_mm=bed_px_mm,
        dz_mm=layer_height_mm,
    )
    return prediction, {"center_mm": center, "scale_mm": scale, "height_mm": height_mm}


def point_cloud_to_gcode(
    model,
    points: np.ndarray,
    out_gcode: str,
    *,
    device,
    gplan_dir: Optional[str] = None,
    layer_height_mm: float = 0.2,
    bed_px_mm: float = 0.2,
    raster_size: Tuple[int, int] = RASTER_SIZE,
    slice_batch_size: int = 8,
    infill_pattern: Optional[str] = "grid",
    infill_scale: float = 1.0,
    gcode_config: Optional[GCodeConfig] = None,
    sample_id: str = "sample",
) -> str:
    """Run the full pipeline of Figure 2 on one point cloud.

    Args:
        points: (N, 3) point cloud in millimetres.
        out_gcode: destination ``.gcode`` file.
        gplan_dir: where to keep the intermediate G-plan map; a temporary
            directory is used when omitted.
        infill_pattern: template to synthesise inside the predicted infill
            regions; None keeps the raw predicted infill.

    Returns:
        The path of the written G-code file.
    """
    import tempfile

    prediction, _ = predict_from_points(
        model,
        points,
        device=device,
        layer_height_mm=layer_height_mm,
        bed_px_mm=bed_px_mm,
        raster_size=raster_size,
        slice_batch_size=slice_batch_size,
    )

    M, R = prediction.M, prediction.R
    if infill_pattern is not None and R is not None:
        M, R = apply_infill(M, R, pattern=infill_pattern, scale=infill_scale, device=device)

    with tempfile.TemporaryDirectory() as tmp:
        folder = gplan_dir or tmp
        save_gplan_folder(
            folder, M, R, prediction.Q,
            px_mm=prediction.px_mm, dz_mm=prediction.dz_mm, sample_id=sample_id,
        )
        return gplan_to_gcode(folder, out_gcode, gcode_config or GCodeConfig())
