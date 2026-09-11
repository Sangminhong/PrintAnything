#!/usr/bin/env python3
"""Print your own point cloud: point cloud file -> G-code.

    python tools/infer.py --ckpt runs/printanything/checkpoints/ep049.pt \
        --input scan.ply --out scan.gcode --bed_center_xy_mm 110,110

Accepts ``.ply``, ``.xyz``/``.txt`` (the first three columns are used), ``.npy``
of shape (N, 3), and mesh files that trimesh can read (which are sampled). The
cloud is expected in millimetres; ``--scale_to_mm`` rescales the object so that
its largest dimension matches a given size, which is the usual way to fit a
scanned object onto the bed.

Normals are never used -- that is the point of the method.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import add_gcode_args, gcode_config_from_args  # noqa: E402
from printanything.infill import PATTERNS  # noqa: E402
from printanything.inference import point_cloud_to_gcode  # noqa: E402
from printanything.models import load_gpnet_checkpoint  # noqa: E402
from printanything.utils import resolve_checkpoint, set_seed  # noqa: E402


def load_points(path: str, num_points: int) -> np.ndarray:
    """Read a point cloud (or sample one from a mesh) as (N, 3) float64 mm."""
    suffix = Path(path).suffix.lower()

    if suffix == ".npy":
        return np.asarray(np.load(path), dtype=np.float64).reshape(-1, 3)

    if suffix in (".xyz", ".txt", ".pwn", ".csv"):
        data = np.loadtxt(path, delimiter="," if suffix == ".csv" else None)
        return np.asarray(data, dtype=np.float64).reshape(len(data), -1)[:, :3]

    import trimesh

    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.PointCloud):
        return np.asarray(loaded.vertices, dtype=np.float64)
    if isinstance(loaded, trimesh.Scene):
        if len(loaded.geometry) == 0:
            raise ValueError(f"{path} contains no geometry")
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    if isinstance(loaded, trimesh.Trimesh):
        points, _ = trimesh.sample.sample_surface(loaded, num_points)
        return np.asarray(points, dtype=np.float64)
    return np.asarray(loaded.vertices, dtype=np.float64)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_gcode_args(ap)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--input", required=True, help="Point cloud (or mesh) file.")
    ap.add_argument("--out", required=True, help="Destination .gcode file.")
    ap.add_argument("--gplan_dir", default="", help="Also keep the predicted G-plan map here.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num_points", type=int, default=30000,
                    help="Points to sample when the input is a mesh.")
    ap.add_argument("--layer_height_mm", type=float, default=0.2)
    ap.add_argument("--pixel_size_mm", type=float, default=0.2,
                    help="Physical size of one G-plan pixel.")
    ap.add_argument("--raster", type=int, default=256, help="G-plan map resolution.")
    ap.add_argument("--slice_batch_size", type=int, default=8)
    ap.add_argument("--scale_to_mm", type=float, default=0.0,
                    help="Rescale the object so its largest dimension is this many mm (0 = keep).")
    ap.add_argument("--infill_pattern", default="grid", choices=("", *PATTERNS))
    ap.add_argument("--infill_scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    points = load_points(args.input, args.num_points)
    if points.size == 0:
        sys.exit(f"No points read from {args.input}")

    extent = points.max(axis=0) - points.min(axis=0)
    if args.scale_to_mm > 0:
        largest = float(extent.max())
        if largest > 1e-9:
            center = 0.5 * (points.max(axis=0) + points.min(axis=0))
            points = (points - center) * (args.scale_to_mm / largest) + center
            extent = points.max(axis=0) - points.min(axis=0)

    print(f"{points.shape[0]} points | bounding box {extent[0]:.1f} x {extent[1]:.1f} x {extent[2]:.1f} mm")

    model, _ = load_gpnet_checkpoint(resolve_checkpoint(args.ckpt), map_location="cpu")
    model = model.to(device).eval()

    out_path = point_cloud_to_gcode(
        model,
        points,
        args.out,
        device=device,
        gplan_dir=args.gplan_dir or None,
        layer_height_mm=args.layer_height_mm,
        bed_px_mm=args.pixel_size_mm,
        raster_size=(args.raster, args.raster),
        slice_batch_size=args.slice_batch_size,
        infill_pattern=args.infill_pattern or None,
        infill_scale=args.infill_scale,
        gcode_config=gcode_config_from_args(args),
        sample_id=Path(args.input).stem,
    )
    print(f"[done] wrote {out_path}")
    print("Preview it in a slicer before printing; the G-code is emitted for the "
          "printer profile you passed on the command line.")


if __name__ == "__main__":
    main()
