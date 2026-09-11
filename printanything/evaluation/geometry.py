"""Geometric metrics (paper, Sec. 5.2).

The predicted G-plan map is turned back into 3D points in millimetres and
compared against points sampled on the ground-truth mesh:

    CD      symmetric Chamfer distance, Eq. (9)
    F1_3D   point F1 at a distance threshold tau = 1 mm, Eq. (10)
    F1_2D   binary occupancy F1 per rasterised slice, averaged, Eq. (11)

The predicted side uses the *perimeter* of the occupancy map rather than every
occupied pixel, since the surface of the print is what the mesh samples describe.
"""

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..gplan.constants import SKIRT

__all__ = [
    "RasterFrame",
    "mesh_surface_points",
    "occupancy_points",
    "perimeter_points",
    "region_points",
    "chamfer_distance",
    "point_f1",
    "slice_iou_f1",
    "per_slice_point_f1",
    "point_inside_validity",
    "align_isotropic",
    "align_prediction_to_mesh",
]


@dataclass
class RasterFrame:
    """Mapping from G-plan pixels to physical millimetres.

    A slice is stored on a canvas that may have been downsampled by
    ``scale_to_target`` and centre-padded by ``(pad_ox, pad_oy)``, so a target
    pixel ``(row, col)`` maps back to the printer frame as::

        x_mm = origin_x + (col - pad_ox + 0.5) / scale * px_mm
        y_mm = origin_y + (row - pad_oy + 0.5) / scale * px_mm
        z_mm = z_start_mm + z_index * dz_mm
    """

    origin_xy_mm: Tuple[float, float] = (0.0, 0.0)
    px_mm: float = 0.2
    dz_mm: float = 0.2
    scale_to_target: float = 1.0
    pad_ox: int = 0
    pad_oy: int = 0
    z_start_mm: float = 0.0

    def to_mm(self, rows, cols, z_index) -> np.ndarray:
        """Vectorised pixel -> mm for arrays of rows / cols / slice indices."""
        s = float(self.scale_to_target) if abs(self.scale_to_target) > 1e-9 else 1.0
        x = self.origin_xy_mm[0] + (np.asarray(cols, np.float64) - self.pad_ox + 0.5) / s * self.px_mm
        y = self.origin_xy_mm[1] + (np.asarray(rows, np.float64) - self.pad_oy + 0.5) / s * self.px_mm
        z = self.z_start_mm + np.asarray(z_index, np.float64) * self.dz_mm
        return np.stack([x, y, np.broadcast_to(z, x.shape)], axis=-1)


def _as_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def mesh_surface_points(stl_path: str, num_points: int = 50000) -> np.ndarray:
    """Uniformly sample the ground-truth mesh surface, in mm."""
    import trimesh

    mesh = trimesh.load(stl_path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        if isinstance(mesh, trimesh.Scene) and len(mesh.geometry) > 0:
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        else:
            return np.zeros((0, 3), dtype=np.float64)
    try:
        pts, _ = trimesh.sample.sample_surface(mesh, num_points)
        return np.asarray(pts, dtype=np.float64)
    except Exception:
        return np.zeros((0, 3), dtype=np.float64)


def occupancy_points(M, frame: RasterFrame) -> np.ndarray:
    """Every occupied pixel of a (Z, H, W) occupancy map, as (N, 3) mm points."""
    M = (_as_numpy(M) > 0).astype(np.uint8)
    chunks = []
    for z in range(M.shape[0]):
        idx = np.argwhere(M[z] > 0)
        if idx.size:
            chunks.append(frame.to_mm(idx[:, 0], idx[:, 1], z))
    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 3), dtype=np.float64)


def perimeter_points(
    M,
    frame: RasterFrame,
    R=None,
    exclude_class: Optional[int] = SKIRT,
    longest_only: bool = True,
) -> np.ndarray:
    """Contours of every slice, as (N, 3) mm points.

    When ``R`` is given, pixels of ``exclude_class`` are removed first, which
    drops the skirt -- a loop printed around the object that is not part of its
    geometry and would otherwise dominate the contour.

    ``longest_only`` keeps only the longest contour of each slice, treating it as
    the outer boundary. That is what the reported numbers were computed with, so
    it stays the default, but it under-represents objects whose slices fall into
    several islands: every island but the largest is dropped on the prediction
    side, while the ground truth still samples the whole mesh. Pass
    ``longest_only=False`` to keep every contour.
    """
    from skimage import measure

    M = (_as_numpy(M) > 0).astype(np.uint8)
    R = None if R is None else _as_numpy(R).astype(np.uint8)

    chunks = []
    for z in range(M.shape[0]):
        mask = M[z]
        if R is not None and exclude_class is not None:
            mask = (mask > 0) & (R[z] != exclude_class)
            mask = mask.astype(np.uint8)
        if mask.sum() == 0:
            continue
        contours = measure.find_contours(mask, level=0.5)
        if not contours:
            continue
        if longest_only:
            contours = [max(contours, key=len)]
        for contour in contours:  # (N, 2) as (row, col)
            chunks.append(frame.to_mm(contour[:, 0], contour[:, 1], z))

    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 3), dtype=np.float64)


def region_points(M, R, region_class: int, frame: RasterFrame) -> np.ndarray:
    """Occupied pixels of one region class, as (N, 3) mm points."""
    M = (_as_numpy(M) > 0).astype(np.uint8)
    R = _as_numpy(R).astype(np.uint8)
    chunks = []
    for z in range(M.shape[0]):
        idx = np.argwhere((M[z] > 0) & (R[z] == region_class))
        if idx.size:
            chunks.append(frame.to_mm(idx[:, 0], idx[:, 1], z))
    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 3), dtype=np.float64)


def _nearest_distances(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    from scipy.spatial import cKDTree

    return cKDTree(b).query(a, k=1)[0], cKDTree(a).query(b, k=1)[0]


def chamfer_distance(pred_pts: np.ndarray, gt_pts: np.ndarray) -> float:
    """Symmetric Chamfer distance in mm, Eq. (9)."""
    pred = np.asarray(pred_pts, np.float64).reshape(-1, 3)
    gt = np.asarray(gt_pts, np.float64).reshape(-1, 3)
    if pred.size == 0 or gt.size == 0:
        return float("nan")
    d_pred, d_gt = _nearest_distances(pred, gt)
    return float(np.mean(d_pred) + np.mean(d_gt))


def point_f1(pred_pts: np.ndarray, gt_pts: np.ndarray, tau_mm: float = 1.0) -> Tuple[float, float, float]:
    """Precision / recall / F1 of matched points at threshold ``tau_mm``, Eq. (10)."""
    pred = np.asarray(pred_pts, np.float64).reshape(-1, 3)
    gt = np.asarray(gt_pts, np.float64).reshape(-1, 3)
    if pred.size == 0 and gt.size == 0:
        return 1.0, 1.0, 1.0
    if pred.size == 0 or gt.size == 0:
        return 0.0, 0.0, 0.0

    d_pred, d_gt = _nearest_distances(pred, gt)
    precision = float(np.mean(d_pred <= tau_mm))
    recall = float(np.mean(d_gt <= tau_mm))
    f1 = 0.0 if precision + recall <= 0 else 2.0 * precision * recall / (precision + recall)
    return precision, recall, f1


def slice_iou_f1(M_pred, M_gt) -> Tuple[float, float]:
    """Mean IoU and mean binary F1 of the occupancy maps over slices, Eq. (11)."""
    M_pred = (_as_numpy(M_pred) > 0).astype(np.float64)
    M_gt = (_as_numpy(M_gt) > 0).astype(np.float64)

    if M_pred.shape != M_gt.shape:
        import torch
        import torch.nn.functional as F

        t = torch.from_numpy(M_gt).float().unsqueeze(0)
        t = F.interpolate(t, size=(M_pred.shape[1], M_pred.shape[2]), mode="nearest")
        M_gt = (t[0].numpy() > 0.5).astype(np.float64)

    ious, f1s = [], []
    for z in range(M_pred.shape[0]):
        p, t = M_pred[z], M_gt[z]
        inter = float((p * t).sum())
        union = float((p + t - p * t).sum())
        ious.append(1.0 if union < 1e-9 and inter < 1e-9 else (inter / union if union >= 1e-9 else 0.0))

        p_sum, t_sum = float(p.sum()), float(t.sum())
        if p_sum < 1e-9 and t_sum < 1e-9:
            f1s.append(1.0)
        elif p_sum < 1e-9 or t_sum < 1e-9:
            f1s.append(0.0)
        else:
            prec, rec = inter / p_sum, inter / t_sum
            f1s.append(0.0 if prec + rec < 1e-9 else 2.0 * prec * rec / (prec + rec))

    return float(np.mean(ious)), float(np.mean(f1s))


def per_slice_point_f1(
    pred_pts: np.ndarray, gt_pts: np.ndarray, num_slices: int, tau_mm: float = 1.0
) -> float:
    """Point F1 computed inside each z-bin and averaged over bins.

    Points are binned by height over the GT z-range, so a prediction that is
    globally close but shifted along z is penalised.
    """
    pred = np.asarray(pred_pts, np.float64).reshape(-1, 3)
    gt = np.asarray(gt_pts, np.float64).reshape(-1, 3)
    Z = int(num_slices)
    if Z <= 0:
        return float("nan")
    if pred.size == 0 and gt.size == 0:
        return 1.0

    reference = gt if gt.size > 0 else pred
    if reference.size == 0:
        return float("nan")
    z_min, z_max = float(reference[:, 2].min()), float(reference[:, 2].max())
    z_range = max(z_max - z_min, 1e-9)

    def bin_index(points):
        if points.size == 0:
            return np.zeros((0,), dtype=np.int64)
        return np.clip(np.round((points[:, 2] - z_min) / z_range * (Z - 1)).astype(np.int64), 0, Z - 1)

    pred_bin, gt_bin = bin_index(pred), bin_index(gt)
    empty = np.zeros((0, 3), dtype=np.float64)

    f1s = []
    for z in range(Z):
        p = pred[pred_bin == z] if pred.size else empty
        t = gt[gt_bin == z] if gt.size else empty
        f1s.append(point_f1(p, t, tau_mm=tau_mm)[2])
    return float(np.mean(f1s))


def point_inside_validity(points_mm: np.ndarray, stl_path: str) -> float:
    """Fraction of points that fall inside the GT mesh (NaN if not computable).

    Used to check that predicted infill stays inside the object. Needs ``rtree``
    for the ray-based containment test.
    """
    import importlib.util

    points_mm = np.asarray(points_mm, np.float64).reshape(-1, 3)
    if points_mm.size == 0 or not stl_path or not os.path.isfile(stl_path):
        return float("nan")
    if importlib.util.find_spec("rtree") is None:
        return float("nan")

    import trimesh

    mesh = trimesh.load(stl_path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        if isinstance(mesh, trimesh.Scene) and len(mesh.geometry) > 0:
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        else:
            return float("nan")

    try:
        return float(np.mean(mesh.contains(points_mm)))
    except Exception:
        try:
            from trimesh.proximity import signed_distance

            return float(np.mean(signed_distance(mesh, points_mm) >= -1e-6))
        except Exception:
            return float("nan")


def align_isotropic(pred_pts: np.ndarray, gt_pts: np.ndarray) -> Tuple[np.ndarray, float]:
    """Align a prediction to the GT by bounding-box centre and diagonal length.

    Predictions live on the printer bed while the mesh lives in its own frame, so
    a single isotropic scale plus a translation puts them in the same frame
    without rotating anything (the slicing axis must be preserved).

    Returns:
        ``(pred_aligned, scale)``.
    """
    pred = np.asarray(pred_pts, np.float64).reshape(-1, 3)
    gt = np.asarray(gt_pts, np.float64).reshape(-1, 3)
    if pred.size == 0 or gt.size == 0:
        return pred, 1.0

    p_min, p_max = pred.min(axis=0), pred.max(axis=0)
    g_min, g_max = gt.min(axis=0), gt.max(axis=0)
    p_diag = float(np.linalg.norm(p_max - p_min))
    g_diag = float(np.linalg.norm(g_max - g_min))
    scale = 1.0 if (p_diag < 1e-9 or g_diag < 1e-9) else g_diag / p_diag
    aligned = (pred - 0.5 * (p_min + p_max)) * scale + 0.5 * (g_min + g_max)
    return aligned, float(scale)


def align_prediction_to_mesh(
    pred_pts: np.ndarray,
    gt_pts: np.ndarray,
    frame: RasterFrame,
    raster_H: int,
    raster_W: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Move predicted points from the printer frame into the mesh frame.

    The slicer recentres the object on the bed, so the two frames differ by a
    translation: the XY centre of the raster is matched to the XY bounding-box
    centre of the mesh -- the same correspondence the slice projection uses -- and
    the bottom slice is matched to the lowest point of the mesh. No scaling and no
    rotation are applied, so the metrics stay sensitive to size errors.

    Returns:
        ``(pred_in_mesh_frame, shift)`` where ``shift`` is the applied (x, y, z).
    """
    pred = np.asarray(pred_pts, np.float64).reshape(-1, 3).copy()
    gt = np.asarray(gt_pts, np.float64).reshape(-1, 3)
    if pred.size == 0 or gt.size == 0:
        return pred, np.zeros(3)

    raster_cx = frame.origin_xy_mm[0] + 0.5 * raster_W * frame.px_mm
    raster_cy = frame.origin_xy_mm[1] + 0.5 * raster_H * frame.px_mm
    mesh_cx = 0.5 * (float(gt[:, 0].min()) + float(gt[:, 0].max()))
    mesh_cy = 0.5 * (float(gt[:, 1].min()) + float(gt[:, 1].max()))

    shift = np.array([mesh_cx - raster_cx, mesh_cy - raster_cy, 0.0], dtype=np.float64)
    pred += shift
    z_shift = float(gt[:, 2].min()) - float(pred[:, 2].min())
    pred[:, 2] += z_shift
    shift[2] = z_shift
    return pred, shift
