"""Slice-100K dataset: point cloud -> ground-truth G-plan map.

Each sample pairs a surface point cloud sampled from the CAD model (STL) with the
ground-truth G-plan map (occupancy M, region R, flow Q) rasterised from the
reference G-code by ``printanything.gplan.gcode_to_gplan``.

The rasters are resized "downsample-to-fit and centre-pad" onto a fixed
``target_h x target_w`` canvas -- never cropped, so no geometry is lost -- and the
exact raw-to-target mapping is returned with every sample so that the slice-wise
point projection can be aligned pixel-for-pixel with the supervision.
"""
import hashlib
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from scipy import ndimage
from torch.utils.data import Dataset

from ..gplan.io import find_manifest


class GPlanDataset(Dataset):
    """
    Directory layout::

        <stl_root>/STLs_*/<mesh_id>.stl
        <gplan_root>/<sample_id>/
            gplan_manifest.json          (``ir_manifest.json`` is also accepted)
            layers/L000_M.npy            occupancy
            layers/L000_R.npy            region
            layers/L000_Q.npy            flow

    ``__getitem__`` returns a dict with:
        id           str, sample identifier
        pc           (N, 3) float32 point cloud, normalised to [-1, 1]
        M            (Z, H, W) uint8 occupancy
        R            (Z, H, W) uint8 region        (if ``use_R``)
        Q            (Z, H, W) float32 flow        (if ``use_Q``)
        px_mm_label  float, pixel size in mm
        dz_mm_label  float, layer height in mm
        ctr_mm       (3,) centre used to normalise the point cloud
        scale_mm     float, scale used to normalise the point cloud
        raster_*     printer-frame raster metadata (see ``_load_gplan_stack``)
    """

    def __init__(
        self,
        stl_root: str = "data/slice100k/stls",
        gplan_root: str = "data/slice100k_gplan",
        num_points: int = 30000,
        use_R: bool = True,
        use_Q: bool = False,
        normalize_pc: bool = True,
        scale_to: float = 1.0,
        cache_index: bool = True,
        cache_path: Optional[str] = None,
        target_h: Optional[int] = None,
        target_w: Optional[int] = None,
        q_cache_root: Optional[str] = None,
        max_objects: Optional[int] = None,
    ):
        """
        Args:
            stl_root: Root directory containing the STL meshes
            gplan_root: Root directory containing the ground-truth G-plan maps
            num_points: Number of points to sample from mesh
            use_R: Whether to load region rasters
            use_Q: Whether to load flow rasters
            normalize_pc: Whether to normalize point cloud
            scale_to: Scale factor for normalization (default 1.0 for [-1, 1])
            cache_index: Whether to cache the dataset index
            cache_path: Path to the index cache file (auto-generated if None)
            q_cache_root: Directory for the cached normalised flow maps; the
                normalisation is expensive, so caching it speeds up training a
                lot. None disables caching.
            max_objects: Keep only the first N indexed objects. The runs reported
                in the paper used the first 1000; see docs/REPRODUCE.md. None
                uses every object that has both a mesh and a G-plan map.
            target_h: Target height for raster (None = use original size)
            target_w: Target width for raster (None = use original size)
        """
        super().__init__()
        self.stl_root = stl_root
        self.gplan_root = gplan_root
        self.num_points = num_points
        self.use_R = use_R
        self.use_Q = use_Q
        self.normalize_pc = normalize_pc
        self.scale_to = scale_to
        self.target_h = target_h
        self.target_w = target_w
        self.q_cache_root = q_cache_root

        if cache_index:
            if cache_path is None:
                cache_path = os.path.join(os.getcwd(), ".cache", "gplan_index.json")
            self.samples = self._load_or_build_index(cache_path)
        else:
            self.samples = self._build_index()

        indexed = len(self.samples)
        if max_objects:
            self.samples = self.samples[: int(max_objects)]
        if len(self.samples) == indexed:
            print(f"[GPlanDataset] {indexed} objects")
        else:
            print(f"[GPlanDataset] {len(self.samples)} of {indexed} indexed objects "
                  f"(--max_objects)")

    def _normalize_flow_continuous(self, Q: np.ndarray, target_range=(0.0, 1.0), cache_path: Optional[str] = None) -> np.ndarray:
        """
        Normalize flow Q to a continuous representation in target_range.
        Converts sparse Q (only toolpath pixels) to smooth continuous field.
        Optionally caches the result to avoid recomputation.

        Args:
            Q: (H, W) flow array (sparse, mostly zeros)
            target_range: (min, max) target range for normalized Q
            cache_path: Optional path to cache file. If provided, checks cache first and saves result.

        Returns:
            Normalized Q in target_range, smoothed to be continuous
        """
        # Check cache first
        if cache_path and os.path.exists(cache_path):
            try:
                cached = np.load(cache_path).astype(np.float32)
                # Verify shape matches
                if cached.shape == Q.shape:
                    return cached
            except Exception as e:
                # If cache read fails, recompute
                print(f"Warning: Failed to load Q cache from {cache_path}: {e}")

        q_nonzero = Q[Q > 0]
        if len(q_nonzero) == 0:
            result = np.zeros_like(Q)
        else:
            # Step 1: Normalize by percentile (robust to outliers)
            q_percentile_95 = np.percentile(q_nonzero, 95) if len(q_nonzero) > 0 else Q.max()
            if q_percentile_95 > 0:
                Q_norm = np.clip(Q / q_percentile_95, 0.0, 1.0)
            else:
                Q_norm = Q.copy()

            # Step 2: Apply Gaussian smoothing to create continuous field
            # This spreads flow values around toolpath
            sigma = 2.0  # pixels - adjust for desired smoothness
            Q_smooth = ndimage.gaussian_filter(Q_norm, sigma=sigma)

            # Step 3: Scale to target range
            q_min, q_max = target_range
            if Q_smooth.max() > 0:
                Q_smooth = Q_smooth / Q_smooth.max() * (q_max - q_min) + q_min

            result = Q_smooth.astype(np.float32)

        # Save to cache if path provided
        if cache_path:
            try:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                np.save(cache_path, result)
            except OSError:
                pass  # caching is best effort

        return result

    def _build_index(self) -> List[Dict[str, str]]:
        """
        Index every G-plan folder that has a matching STL mesh.

        Returns list of dicts with keys: id, mesh_id, stl_path, gplan_path
        """
        samples = []

        # The G-plan folders are the source of truth: a mesh without
        # supervision is skipped.
        if not os.path.isdir(self.gplan_root):
            print(f"[GPlanDataset] Warning: G-plan root not found: {self.gplan_root}")
            return samples

        for gplan_name in os.listdir(self.gplan_root):
            gplan_path = os.path.join(self.gplan_root, gplan_name)
            if not os.path.isdir(gplan_path):
                continue

            if find_manifest(gplan_path) is None:
                continue

            # Sample folders are named '<mesh_id>_objaverse_xl_config_<N>'
            # Format: [id]_objaverse_xl_config_[N]
            # Example: 100072_objaverse_xl_config_4
            parts = gplan_name.split("_objaverse_xl_config_")
            if len(parts) != 2:
                continue

            mesh_id = parts[0]

            # Find corresponding STL file
            stl_path = self._find_stl_file(mesh_id)
            if stl_path is None:
                continue

            samples.append({
                "id": gplan_name,
                "mesh_id": mesh_id,
                "stl_path": stl_path,
                "gplan_path": gplan_path,
            })

        return samples

    def _find_stl_file(self, mesh_id: str) -> Optional[str]:
        """
        Find STL file by numeric ID across all STLs_* subdirectories.
        """
        if not os.path.isdir(self.stl_root):
            return None

        stl_filename = f"{mesh_id}.stl"

        # Search in all STLs_* subdirectories
        for subdir in os.listdir(self.stl_root):
            subdir_path = os.path.join(self.stl_root, subdir)
            if not os.path.isdir(subdir_path):
                continue

            stl_path = os.path.join(subdir_path, stl_filename)
            if os.path.exists(stl_path):
                return stl_path

        return None

    def _load_or_build_index(self, cache_path: str) -> List[Dict[str, str]]:
        """Load index from cache or build if missing/stale."""
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    cached = json.load(f)
                # Validate cache entries still exist
                valid = []
                for entry in cached:
                    if os.path.exists(entry["stl_path"]) and os.path.exists(entry["gplan_path"]):
                        valid.append(entry)
                if len(valid) > 0:
                    print(f"[GPlanDataset] Loaded {len(valid)} samples from cache: {cache_path}")
                    return valid
            except Exception as e:
                print(f"[GPlanDataset] Cache load failed: {e}, rebuilding...")

        # Build fresh index
        samples = self._build_index()

        # Save cache
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(samples, f, indent=2)
            print(f"[GPlanDataset] Saved index cache: {cache_path}")
        except Exception as e:
            print(f"[GPlanDataset] Cache save failed: {e}")

        return samples

    def __len__(self) -> int:
        return len(self.samples)


    def _load_pc(self, stl_path: str) -> np.ndarray:
        """Load point cloud from STL mesh."""
        mesh = trimesh.load(stl_path, force="mesh", process=False)
        if not isinstance(mesh, trimesh.Trimesh):
            if isinstance(mesh, trimesh.Scene) and len(mesh.geometry) > 0:
                mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
            else:
                raise ValueError(f"Failed to load mesh from: {stl_path}")

        # Sample points from surface
        points, _ = trimesh.sample.sample_surface(mesh, self.num_points)
        return points.astype(np.float32)

    def _normalize_points(self, pc: np.ndarray, scale_to: float = 1.0) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Normalize point cloud to fit in [-scale_to, scale_to].

        Returns:
            pc_normalized: [N, 3] normalized points
            center: [3] center of original point cloud
            scale: float, scale factor used
        """
        if len(pc) == 0:
            return pc, np.zeros(3, dtype=np.float32), 1.0

        center = pc.mean(axis=0)
        pc_centered = pc - center
        scale = np.abs(pc_centered).max()
        if scale < 1e-6:
            scale = 1.0

        pc_normalized = (pc_centered / scale) * scale_to

        return pc_normalized, center, scale

    def _q_cache_path(self, q_path: str) -> Optional[str]:
        """Cache file for the normalised flow map of ``q_path`` (None = no cache)."""
        if not self.q_cache_root:
            return None
        try:
            key = os.path.relpath(q_path, self.gplan_root)
        except ValueError:
            key = "abs_" + hashlib.md5(q_path.encode()).hexdigest()[:16]
        key = key.replace(os.sep, "_").replace("/", "_").replace("\\", "_")
        return os.path.join(self.q_cache_root, key.replace(".npy", "_normalized.npy"))

    def _load_gplan_stack(self, gplan_path: str):
        """
        Load and stack the per-slice G-plan maps of one sample.

        Returns:
            M: [Z, H, W] occupancy (uint8)
            R: [Z, H, W] region (uint8) or None
            Q: [Z, H, W] flow (float32) or None
            px_mm: pixel size in mm
            dz_mm: average layer height in mm
            meta: dict with raw raster frame info (origin/px/H/W) and the resize->pad mapping to target
        """
        manifest_path = find_manifest(gplan_path)
        if manifest_path is None:
            raise FileNotFoundError(f"No G-plan manifest under {gplan_path}")
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        layers = manifest["layers"]
        if len(layers) == 0:
            raise ValueError(f"No layers in manifest: {manifest_path}")

        # Get raster info from first layer
        first_raster = layers[0]["raster"]
        H = int(first_raster["H"])
        W = int(first_raster["W"])
        px_mm_raw = float(first_raster["px_mm"])
        origin_xy_mm = tuple(first_raster.get("origin_xy_mm", [0.0, 0.0]))
        px_mm = px_mm_raw

        # Compute average layer height
        if len(layers) > 1:
            dz_mm = (layers[-1]["z_mm"] - layers[0]["z_mm"]) / (len(layers) - 1)
        else:
            dz_mm = layers[0]["layer_height_mm"]

        # Use target sizes if specified, otherwise use original sizes.
        # IMPORTANT: We do NOT crop (cropping loses geometry). Instead:
        #   - if a layer is larger than target: downsample-to-fit (no upsample)
        #   - then center-pad into the target canvas
        final_H = self.target_h if self.target_h is not None else H
        final_W = self.target_w if self.target_w is not None else W

        def _resize_to_fit_center_pad(arr: np.ndarray, out_h: int, out_w: int, kind: str):
            """
            kind:
              - 'mask'  : occupancy (binary); downsample with nearest, then threshold
              - 'label' : region labels (uint8); downsample with nearest
              - 'float' : flow (float32); downsample with bilinear
            Returns: (out_arr, scale) where scale<=1 is spatial scale applied.
            """
            h, w = arr.shape
            # Downsample-to-fit (preserve aspect) but never upscale.
            s = min(1.0, out_h / max(h, 1), out_w / max(w, 1))

            if s < 1.0:
                nh = max(1, int(round(h * s)))
                nw = max(1, int(round(w * s)))

                t = torch.from_numpy(arr)
                if kind in ("mask", "label"):
                    t = t.to(torch.float32)[None, None, ...]
                    t = F.interpolate(t, size=(nh, nw), mode="nearest")
                    resized = t[0, 0].cpu().numpy()
                    if kind == "mask":
                        resized = (resized >= 0.5).astype(np.uint8)
                    else:
                        resized = resized.round().astype(np.uint8)
                else:
                    t = t.to(torch.float32)[None, None, ...]
                    t = F.interpolate(t, size=(nh, nw), mode="bilinear", align_corners=False)
                    resized = t[0, 0].cpu().numpy().astype(np.float32)
            else:
                resized = arr

            rh, rw = resized.shape
            out = np.zeros((out_h, out_w), dtype=(np.float32 if kind == "float" else np.uint8))
            oy = (out_h - rh) // 2
            ox = (out_w - rw) // 2
            out[oy:oy + rh, ox:ox + rw] = resized
            return out, s

        # Pre-allocate arrays
        Z = len(layers)
        M_stack = np.zeros((Z, final_H, final_W), dtype=np.uint8)
        R_stack = np.zeros((Z, final_H, final_W), dtype=np.uint8) if self.use_R else None
        Q_stack = np.zeros((Z, final_H, final_W), dtype=np.float32) if self.use_Q else None

        applied_scale = 1.0  # track smallest scale (most downsample) within this sample

        # Load each layer
        for i, layer in enumerate(layers):
            raster = layer["raster"]

            # Load occupancy (M)
            m_path = os.path.join(gplan_path, raster["occupancy_uri"])
            M_layer = np.load(m_path).astype(np.uint8)
            M_out, s = _resize_to_fit_center_pad(M_layer, final_H, final_W, kind="mask")
            M_stack[i] = M_out
            applied_scale = min(applied_scale, s)

            # Load region (R)
            if self.use_R:
                r_path = os.path.join(gplan_path, raster["region_uri"])
                R_layer = np.load(r_path).astype(np.uint8)
                R_out, _ = _resize_to_fit_center_pad(R_layer, final_H, final_W, kind="label")
                R_stack[i] = R_out

            # Load flow (Q)
            if self.use_Q:
                flow_uri = raster.get("flow_uri", None)
                if flow_uri:
                    q_path = os.path.join(gplan_path, flow_uri)
                    Q_layer = np.load(q_path).astype(np.float32)

                    Q_layer = self._normalize_flow_continuous(
                        Q_layer,
                        target_range=(0.0, 1.0),
                        cache_path=self._q_cache_path(q_path),
                    )

                    Q_out, _ = _resize_to_fit_center_pad(Q_layer, final_H, final_W, kind="float")
                    Q_stack[i] = Q_out
                else:
                    Q_stack[i] = 0.0

        # If we downsampled by s (<1), each pixel covers more mm: px_mm' = px_mm / s
        if applied_scale < 1.0:
            px_mm = float(px_mm) / float(applied_scale)

        # Record the deterministic mapping raw(H,W) -> target(final_H,final_W)
        s = min(1.0, final_H / max(H, 1), final_W / max(W, 1))
        rh = max(1, int(round(H * s))) if s < 1.0 else H
        rw = max(1, int(round(W * s))) if s < 1.0 else W
        oy = int((final_H - rh) // 2)
        ox = int((final_W - rw) // 2)
        meta = {
            "origin_xy_mm": (float(origin_xy_mm[0]), float(origin_xy_mm[1])),
            "px_mm_raw": float(px_mm_raw),
            "H_raw": int(H),
            "W_raw": int(W),
            "scale_raw_to_target": float(s),
            "H_resized": int(rh),
            "W_resized": int(rw),
            "pad_oy": int(oy),
            "pad_ox": int(ox),
        }

        return M_stack, R_stack, Q_stack, float(px_mm), float(dz_mm), meta

    def _prepare_point_cloud(self, stl_path: str):
        """Sample the input point cloud of one object.

        Subclasses override this to change how the observation is produced (see
        :class:`~printanything.datasets.scan_simulation.SimulatedScanGPlanDataset`).

        Returns:
            ``(pc_norm, center_mm, scale_mm, extra)`` where ``extra`` holds any
            additional fields to attach to the sample dict.
        """
        pc = self._load_pc(stl_path)  # (N, 3) in mm
        if self.normalize_pc:
            pc_norm, center, scale = self._normalize_points(pc, scale_to=self.scale_to)
        else:
            pc_norm = pc.astype(np.float32)
            center = np.zeros(3, dtype=np.float32)
            scale = 1.0
        return pc_norm, center, scale, {}

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        #sample = self.samples[909]
        sample_id = sample["id"]
        stl_path = sample["stl_path"]
        gplan_path = sample["gplan_path"]

        pc_norm, center, scale, extra = self._prepare_point_cloud(stl_path)

        # Ground-truth G-plan maps + the raster metadata used for alignment
        M, R, Q, px_mm, dz_mm, raster = self._load_gplan_stack(gplan_path)

        # Pack output
        out = {
            "id": sample_id,
            "stl_path": stl_path,
            "pc": torch.from_numpy(pc_norm),  # [N, 3]
            "M": torch.from_numpy(M),  # [Z, H, W]
            "px_mm_label": px_mm,
            "dz_mm_label": dz_mm,
            "ctr_mm": torch.from_numpy(center),  # [3]
            "scale_mm": float(scale),
            # Raw printer-frame raster (before resize/pad), for aligned projection
            "raster_origin_xy_mm": torch.tensor(raster["origin_xy_mm"], dtype=torch.float32),
            "raster_px_mm": float(raster["px_mm_raw"]),
            "raster_H": int(raster["H_raw"]),
            "raster_W": int(raster["W_raw"]),
            # Deterministic mapping raw->target used by the loader
            "raster_scale_to_target": float(raster["scale_raw_to_target"]),
            "raster_pad_oy": int(raster["pad_oy"]),
            "raster_pad_ox": int(raster["pad_ox"]),
        }

        if R is not None:
            out["R"] = torch.from_numpy(R)  # [Z, H, W]

        if Q is not None:
            out["Q"] = torch.from_numpy(Q)  # [Z, H, W]

        out.update(extra)
        return out


