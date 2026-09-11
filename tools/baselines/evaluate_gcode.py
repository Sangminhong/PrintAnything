#!/usr/bin/env python3
"""Evaluate a folder of G-code files on the held-out split (Table 1).

This is how the mesh-based baselines are scored: each pipeline reconstructs a
mesh from the input point cloud, PrusaSlicer turns that mesh into G-code, and the
G-code is rasterised back into a G-plan map so that exactly the same metrics as
``tools/evaluate.py`` apply.

    python tools/baselines/evaluate_gcode.py \
        --gcode_root outputs/poisson/gcode \
        --out_json   outputs/poisson/metrics.json

Reconstruction methods return a mesh in an arbitrary scale, so predictions are
additionally fitted to the ground truth with one isotropic scale before the
metrics are computed (``--no_isotropic_fit`` turns that off); the mean fitted
scale is reported, as it says how far off a baseline's size was.

What is measured here is the rasterised toolpath, which is as wide as the
extrusion width, so features thinner than one line get inflated. That is the same
for every pipeline scored this way, which is the point -- but our own numbers are
produced by ``tools/evaluate.py``, which measures the G-plan map itself.
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from _common import add_data_args, dataset_and_split  # noqa: E402
from printanything.evaluation import (  # noqa: E402
    RasterFrame,
    align_isotropic,
    align_prediction_to_mesh,
    chamfer_distance,
    mesh_surface_points,
    per_slice_point_f1,
    perimeter_points,
    point_f1,
    slice_iou_f1,
)
from printanything.gplan import gcode_to_gplan, load_gplan_stack  # noqa: E402
from printanything.utils import set_seed  # noqa: E402


def find_gcode(sample_id: str, gcode_files: dict):
    """Exact ``<sample_id>.gcode`` first, then any file starting with it."""
    exact = gcode_files.get(f"{sample_id}.gcode")
    if exact:
        return exact
    prefix = f"{sample_id}_"
    for name, path in gcode_files.items():
        if name.startswith(prefix):
            return path
    return None


def rasterize_points(points: np.ndarray, shape_zhw, bounds_xyz) -> np.ndarray:
    """Bin 3D points into a binary (Z, H, W) grid spanning ``bounds_xyz``."""
    Z, H, W = (int(v) for v in shape_zhw)
    out = np.zeros((Z, H, W), dtype=np.uint8)
    pts = np.asarray(points, np.float64).reshape(-1, 3)
    if pts.size == 0 or Z <= 0 or H <= 0 or W <= 0:
        return out

    (min_x, max_x), (min_y, max_y), (min_z, max_z) = bounds_xyz
    j = np.clip(np.round((pts[:, 0] - min_x) / max(max_x - min_x, 1e-9) * (W - 1)), 0, W - 1)
    i = np.clip(np.round((pts[:, 1] - min_y) / max(max_y - min_y, 1e-9) * (H - 1)), 0, H - 1)
    z = np.clip(np.round((pts[:, 2] - min_z) / max(max_z - min_z, 1e-9) * (Z - 1)), 0, Z - 1)
    out[z.astype(np.int64), i.astype(np.int64), j.astype(np.int64)] = 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_args(ap)
    ap.add_argument("--gcode_root", required=True, help="Folder of baseline .gcode files.")
    ap.add_argument("--tau_mm", type=float, default=1.0)
    ap.add_argument("--gt_surface_points", type=int, default=50000)
    ap.add_argument("--all_contours", action="store_true",
                    help="Score every contour of a slice, not only its longest one "
                         "(the reported numbers use the longest only).")
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--no_isotropic_fit", action="store_true",
                    help="Compare in the original scale instead of fitting one isotropic scale.")
    ap.add_argument("--start_after_external_perimeter", action="store_true", default=True,
                    help="Start parsing at the first ';TYPE:External perimeter' marker (PrusaSlicer output).")
    ap.add_argument("--parse_from_first_extrusion", dest="start_after_external_perimeter",
                    action="store_false", help="Parse from the first extrusion instead.")
    ap.add_argument("--out_json", default="")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)

    gcode_root = os.path.abspath(args.gcode_root)
    if not os.path.isdir(gcode_root):
        sys.exit(f"gcode_root not found: {gcode_root}")
    gcode_files = {fn: os.path.join(gcode_root, fn) for fn in os.listdir(gcode_root) if fn.endswith(".gcode")}
    print(f"{len(gcode_files)} G-code files in {gcode_root}")

    dataset, _, val_set = dataset_and_split(args, use_R=True, use_Q=False)
    val_indices = list(val_set.indices)
    if args.max_samples:
        val_indices = val_indices[: args.max_samples]
    print(f"{len(val_indices)} held-out objects")

    scores = {k: [] for k in ("CD", "F1_3D", "F1_2D", "IoU_2D", "fitted_scale")}
    missing, failed = [], []

    for idx in tqdm(val_indices, desc="eval", ncols=100):
        sample = dataset[idx]
        sample_id = sample.get("id", str(idx))

        gcode_path = find_gcode(sample_id, gcode_files)
        if gcode_path is None:
            missing.append(sample_id)
            continue

        try:
            with tempfile.TemporaryDirectory(prefix="gplan_") as tmp:
                manifest = gcode_to_gplan(
                    gcode_path, tmp,
                    start_after_external_perimeter=args.start_after_external_perimeter,
                )
                M_pred, R_pred, _, meta = load_gplan_stack(manifest)

            frame = RasterFrame(
                origin_xy_mm=meta["origin_xy_mm"],
                px_mm=meta["px_mm_raw"],
                dz_mm=meta["dz_mm"],
            )
            pred_pts = perimeter_points(M_pred, frame, R=R_pred, longest_only=not args.all_contours)
            if pred_pts.size == 0:
                pred_pts = perimeter_points(M_pred, frame, R=None, longest_only=not args.all_contours)

            gt_pts = mesh_surface_points(sample["stl_path"], args.gt_surface_points)
            if pred_pts.size == 0 or gt_pts.size == 0:
                failed.append(sample_id)
                continue

            pred_pts, _ = align_prediction_to_mesh(
                pred_pts, gt_pts, frame, raster_H=meta["H_raw"], raster_W=meta["W_raw"]
            )
            fitted_scale = 1.0
            if not args.no_isotropic_fit:
                pred_pts, fitted_scale = align_isotropic(pred_pts, gt_pts)

            center = sample["ctr_mm"].numpy().reshape(3).astype(np.float64)
            scale = max(float(sample["scale_mm"]), 1e-9)
            cd = chamfer_distance((pred_pts - center) / scale, (gt_pts - center) / scale)
            f1_3d = point_f1(pred_pts, gt_pts, tau_mm=args.tau_mm)[2]

            Z, H, W = sample["M"].shape
            f1_2d = per_slice_point_f1(pred_pts, gt_pts, num_slices=Z, tau_mm=args.tau_mm)

            bounds = tuple(
                (float(gt_pts[:, a].min()), float(gt_pts[:, a].max())) for a in range(3)
            )
            iou, _ = slice_iou_f1(
                rasterize_points(pred_pts, (Z, H, W), bounds),
                rasterize_points(gt_pts, (Z, H, W), bounds),
            )

            scores["CD"].append(cd)
            scores["F1_3D"].append(f1_3d)
            scores["F1_2D"].append(f1_2d)
            scores["IoU_2D"].append(iou)
            scores["fitted_scale"].append(fitted_scale)

            if not args.quiet:
                tqdm.write(f"  {sample_id}: CD={cd:.4f} F1_3D={f1_3d:.4f} F1_2D={f1_2d:.4f}")
        except Exception as exc:  # noqa: BLE001
            failed.append(sample_id)
            print(f"  fail {sample_id}: {exc}", file=sys.stderr)

    results = {k: (float(np.nanmean(v)) if v else float("nan")) for k, v in scores.items()}
    results.update(
        num_samples=len(scores["CD"]),
        num_missing_gcode=len(missing),
        num_failed=len(failed),
        gcode_root=gcode_root,
    )

    print("\n=== results ===")
    print(f"CD      {results['CD']:.4f}")
    print(f"F1_3D   {results['F1_3D']:.4f}")
    print(f"F1_2D   {results['F1_2D']:.4f}")
    print(f"IoU_2D  {results['IoU_2D']:.4f}")
    print(f"objects {results['num_samples']} "
          f"(missing G-code {len(missing)}, failed {len(failed)}) "
          f"| mean fitted scale {results['fitted_scale']:.3f}")

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        results["missing"] = missing
        results["failed"] = failed
        with open(args.out_json, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
