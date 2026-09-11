"""G-plan dataset variant whose input is a simulated depth scan.

Same sample contract as :mod:`printanything.datasets.gplan_dataset`, but the
uniform mesh samples are replaced by view-dependent partial scans, which is the
setting of the robustness study in Sec. 5.6. The scanner is dependency-light: it
samples a dense surface cloud, projects it into one or more virtual depth
cameras, keeps the nearest point per pixel, and then optionally adds depth noise
and drops visible pixels.
"""
import math
import random
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import trimesh

from .gplan_dataset import GPlanDataset


def _as_tuple(x, default):
    if x is None:
        return tuple(default)
    if isinstance(x, str):
        vals = [v.strip() for v in x.split(",") if v.strip()]
        return tuple(int(v) for v in vals)
    if isinstance(x, Iterable):
        return tuple(x)
    return (x,)


class SimulatedScanGPlanDataset(GPlanDataset):
    """
    G-plan dataset whose point clouds come from simulated depth scans.

    Args:
        scan_views: Candidate number of virtual views. A value is sampled per
            example when ``randomize_scan`` is true.
        scan_image_size: Virtual depth image resolution used by the z-buffer.
        scan_fov_deg: Pinhole camera field of view.
        scan_points_factor: Dense surface samples before visibility filtering,
            relative to ``num_points``.
        depth_noise_std_rel: Stddev of camera-depth noise as a fraction of the
            mesh normalization scale. This approximates depth sensor noise before
            final point-cloud normalization.
        depth_noise_std_rels: Optional candidate noise levels sampled per example.
        pixel_dropout_prob: Probability of dropping visible depth pixels.
        pixel_dropout_probs: Optional candidate dropout levels sampled per example.
        clean_prob: Probability of falling back to the original uniform mesh
            sampling for a training example.
        normalize_by_full_mesh: If true, normalise partial scans with the full
            mesh centre/scale, which keeps them aligned with the G-plan raster.
    """

    def __init__(
        self,
        *args,
        scan_views: Sequence[int] = (1, 2, 4, 8),
        scan_image_size: int = 256,
        scan_fov_deg: float = 60.0,
        scan_points_factor: float = 8.0,
        depth_noise_std_rel: float = 0.0,
        depth_noise_std_rels: Optional[Sequence[float]] = None,
        pixel_dropout_prob: float = 0.0,
        pixel_dropout_probs: Optional[Sequence[float]] = None,
        clean_prob: float = 0.0,
        randomize_scan: bool = True,
        normalize_by_full_mesh: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.scan_views = tuple(int(v) for v in _as_tuple(scan_views, (1, 2, 4, 8)) if int(v) > 0)
        if not self.scan_views:
            self.scan_views = (4,)
        self.scan_image_size = int(scan_image_size)
        self.scan_fov_deg = float(scan_fov_deg)
        self.scan_points_factor = float(scan_points_factor)
        self.depth_noise_std_rel = float(depth_noise_std_rel)
        self.depth_noise_std_rels = (
            tuple(float(v) for v in depth_noise_std_rels)
            if depth_noise_std_rels is not None
            else None
        )
        self.pixel_dropout_prob = float(pixel_dropout_prob)
        self.pixel_dropout_probs = (
            tuple(float(v) for v in pixel_dropout_probs)
            if pixel_dropout_probs is not None
            else None
        )
        self.clean_prob = float(clean_prob)
        self.randomize_scan = bool(randomize_scan)
        self.normalize_by_full_mesh = bool(normalize_by_full_mesh)

    def _load_mesh(self, stl_path: str) -> trimesh.Trimesh:
        mesh = trimesh.load(stl_path, force="mesh", process=False)
        if not isinstance(mesh, trimesh.Trimesh):
            if isinstance(mesh, trimesh.Scene) and len(mesh.geometry) > 0:
                mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
            else:
                raise ValueError(f"Failed to load mesh from: {stl_path}")
        return mesh

    def _mesh_center_scale(self, mesh: trimesh.Trimesh) -> Tuple[np.ndarray, float]:
        verts = np.asarray(mesh.vertices, dtype=np.float32)
        if verts.size == 0:
            return np.zeros(3, dtype=np.float32), 1.0
        center = verts.mean(axis=0).astype(np.float32)
        scale = float(np.abs(verts - center[None, :]).max())
        if scale < 1e-6:
            scale = 1.0
        return center, scale

    def _sample_scan_settings(self) -> Dict[str, float]:
        views = random.choice(self.scan_views) if self.randomize_scan else self.scan_views[-1]
        if self.depth_noise_std_rels is not None and self.randomize_scan:
            noise_rel = random.choice(self.depth_noise_std_rels)
        else:
            noise_rel = self.depth_noise_std_rel
        if self.pixel_dropout_probs is not None and self.randomize_scan:
            dropout = random.choice(self.pixel_dropout_probs)
        else:
            dropout = self.pixel_dropout_prob
        return {
            "views": int(views),
            "depth_noise_std_rel": float(max(0.0, noise_rel)),
            "pixel_dropout_prob": float(min(max(dropout, 0.0), 0.95)),
        }

    def _camera_directions(self, n_views: int) -> np.ndarray:
        # Fibonacci sphere directions give repeatable, roughly even coverage.
        dirs = []
        golden = math.pi * (3.0 - math.sqrt(5.0))
        for i in range(n_views):
            z = 1.0 - (2.0 * (i + 0.5) / n_views)
            r = math.sqrt(max(0.0, 1.0 - z * z))
            theta = i * golden
            dirs.append([math.cos(theta) * r, math.sin(theta) * r, z])
        return np.asarray(dirs, dtype=np.float32)

    def _visible_points_for_view(
        self,
        points_mm: np.ndarray,
        center_mm: np.ndarray,
        scale_mm: float,
        direction: np.ndarray,
        depth_noise_std_rel: float,
        pixel_dropout_prob: float,
    ) -> np.ndarray:
        direction = direction.astype(np.float32)
        direction /= max(float(np.linalg.norm(direction)), 1e-6)
        cam_pos = center_mm + direction * (3.0 * scale_mm)
        forward = center_mm - cam_pos
        forward /= max(float(np.linalg.norm(forward)), 1e-6)

        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        if abs(float(np.dot(up, forward))) > 0.95:
            up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(forward, up)
        right /= max(float(np.linalg.norm(right)), 1e-6)
        up = np.cross(right, forward)

        rel = points_mm - cam_pos[None, :]
        x = rel @ right
        y = rel @ up
        z = rel @ forward
        valid = z > 1e-6
        if not np.any(valid):
            return np.empty((0, 3), dtype=np.float32)

        f = 0.5 * float(self.scan_image_size) / math.tan(math.radians(self.scan_fov_deg) * 0.5)
        u = (f * (x[valid] / z[valid]) + self.scan_image_size * 0.5).astype(np.int64)
        v = (f * (y[valid] / z[valid]) + self.scan_image_size * 0.5).astype(np.int64)
        z_valid = z[valid]
        idx_valid = np.nonzero(valid)[0]

        in_frame = (u >= 0) & (u < self.scan_image_size) & (v >= 0) & (v < self.scan_image_size)
        if not np.any(in_frame):
            return np.empty((0, 3), dtype=np.float32)

        u = u[in_frame]
        v = v[in_frame]
        z_valid = z_valid[in_frame]
        idx_valid = idx_valid[in_frame]

        pix = v * self.scan_image_size + u
        order = np.lexsort((z_valid, pix))
        pix_sorted = pix[order]
        first = np.ones_like(pix_sorted, dtype=bool)
        first[1:] = pix_sorted[1:] != pix_sorted[:-1]
        visible_idx = idx_valid[order[first]]
        visible = points_mm[visible_idx].astype(np.float32).copy()

        if pixel_dropout_prob > 0:
            keep = np.random.rand(visible.shape[0]) >= float(pixel_dropout_prob)
            visible = visible[keep]

        if depth_noise_std_rel > 0 and visible.shape[0] > 0:
            noise_std = float(depth_noise_std_rel) * float(scale_mm)
            depth_noise = np.random.normal(0.0, noise_std, size=(visible.shape[0], 1)).astype(np.float32)
            visible = visible + forward[None, :].astype(np.float32) * depth_noise

        return visible

    def _resample_to_num_points(self, pc: np.ndarray) -> np.ndarray:
        if pc.shape[0] == 0:
            return np.zeros((self.num_points, 3), dtype=np.float32)
        if pc.shape[0] == self.num_points:
            return pc.astype(np.float32)
        replace = pc.shape[0] < self.num_points
        idx = np.random.choice(pc.shape[0], size=self.num_points, replace=replace)
        return pc[idx].astype(np.float32)

    def _simulate_scan(self, mesh: trimesh.Trimesh, center: np.ndarray, scale: float, settings: Dict[str, float]) -> np.ndarray:
        dense_n = max(self.num_points, int(round(self.num_points * self.scan_points_factor)))
        points, _ = trimesh.sample.sample_surface(mesh, dense_n)
        points = points.astype(np.float32)
        chunks = []
        for direction in self._camera_directions(int(settings["views"])):
            visible = self._visible_points_for_view(
                points,
                center,
                scale,
                direction,
                depth_noise_std_rel=float(settings["depth_noise_std_rel"]),
                pixel_dropout_prob=float(settings["pixel_dropout_prob"]),
            )
            if visible.shape[0] > 0:
                chunks.append(visible)
        if not chunks:
            return self._resample_to_num_points(points)
        return self._resample_to_num_points(np.concatenate(chunks, axis=0))

    def _prepare_point_cloud(self, stl_path: str):
        mesh = self._load_mesh(stl_path)
        mesh_center, mesh_scale = self._mesh_center_scale(mesh)

        if self.clean_prob > 0.0 and random.random() < self.clean_prob:
            pc = self._load_pc(stl_path)
            scan_meta = {
                "scan_type": "clean_uniform",
                "views": 0,
                "depth_noise_std_rel": 0.0,
                "pixel_dropout_prob": 0.0,
            }
        else:
            scan_meta = self._sample_scan_settings()
            scan_meta["scan_type"] = "simulated_depth"
            pc = self._simulate_scan(mesh, mesh_center, mesh_scale, scan_meta)

        if not self.normalize_pc:
            pc_norm, center, scale = pc.astype(np.float32), np.zeros(3, dtype=np.float32), 1.0
        elif self.normalize_by_full_mesh:
            center, scale = mesh_center, mesh_scale
            pc_norm = ((pc - center[None, :]) / scale * self.scale_to).astype(np.float32)
        else:
            pc_norm, center, scale = self._normalize_points(pc, scale_to=self.scale_to)

        extra = {
            "scan_type": scan_meta["scan_type"],
            "scan_views": int(scan_meta["views"]),
            "scan_depth_noise_std_rel": float(scan_meta["depth_noise_std_rel"]),
            "scan_pixel_dropout_prob": float(scan_meta["pixel_dropout_prob"]),
        }
        return pc_norm.astype(np.float32), center.astype(np.float32), float(scale), extra
