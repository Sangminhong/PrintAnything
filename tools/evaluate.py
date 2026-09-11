#!/usr/bin/env python3
"""Evaluate PrintAnything on the held-out split of Slice-100K (Sec. 5.2).

    python tools/evaluate.py --ckpt runs/printanything/checkpoints/ep049.pt

Reported metrics, averaged over the test objects:

    CD        Chamfer distance between the predicted print surface and the GT
              mesh, in normalised units (the object is scaled to [-1, 1] first),
              Eq. (9).
    F1_3D     point F1 at tau = 1 mm, Eq. (10).
    F1_2D     the same point F1 computed inside each slice and then averaged over
              slices, Eq. (11); it is what exposes slice-level errors that the
              global F1 hides.
    IoU_2D    IoU of the predicted occupancy raster against the GT raster.

The predicted side is the perimeter of the occupancy map (the surface that gets
printed), with skirt pixels removed; the GT side is a uniform sampling of the
reference mesh.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import add_data_args, dataset_and_split  # noqa: E402
from printanything.evaluation import (  # noqa: E402
    RasterFrame,
    align_prediction_to_mesh,
    chamfer_distance,
    gplan_flow_metrics,
    mesh_surface_points,
    occupancy_points,
    per_slice_point_f1,
    perimeter_points,
    point_f1,
    slice_iou_f1,
)
from printanything.inference import predict_gplan  # noqa: E402
from printanything.models import load_gpnet_checkpoint  # noqa: E402
from printanything.utils import resolve_checkpoint, set_seed  # noqa: E402


def frame_from_sample(sample) -> RasterFrame:
    origin = sample.get("raster_origin_xy_mm", torch.zeros(2))
    origin = origin.detach().cpu().numpy() if torch.is_tensor(origin) else np.asarray(origin)
    return RasterFrame(
        origin_xy_mm=(float(origin[0]), float(origin[1])),
        px_mm=float(sample.get("raster_px_mm", 0.2)),
        dz_mm=float(sample.get("dz_mm_label", 0.2)),
        scale_to_target=float(sample.get("raster_scale_to_target", 1.0)),
        pad_ox=int(sample.get("raster_pad_ox", 0)),
        pad_oy=int(sample.get("raster_pad_oy", 0)),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_args(ap)
    ap.add_argument("--ckpt", required=True, help="Trained GPNet checkpoint.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--slice_batch_size", type=int, default=8)
    ap.add_argument("--tau_mm", type=float, default=1.0, help="Distance threshold of the point F1.")
    ap.add_argument("--gt_surface_points", type=int, default=50000)
    ap.add_argument("--all_contours", action="store_true",
                    help="Score every contour of a slice, not only its longest one "
                         "(the reported numbers use the longest only).")
    ap.add_argument("--max_samples", type=int, default=0, help="Evaluate only the first N objects.")
    ap.add_argument("--out_json", default="", help="Write the metrics to this file.")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    model, ckpt = load_gpnet_checkpoint(resolve_checkpoint(args.ckpt), map_location="cpu")
    model = model.to(device).eval()
    print(f"checkpoint: {args.ckpt} (epoch {ckpt.get('epoch', ckpt.get('ep', '?'))}, "
          f"encoder {model.point_encoder_name})")

    _, _, val_set = dataset_and_split(args, use_R=model.predict_R, use_Q=model.predict_Q)
    indices = range(len(val_set) if not args.max_samples else min(args.max_samples, len(val_set)))
    print(f"evaluating {len(indices)} held-out objects")

    scores = {k: [] for k in ("CD", "F1_3D", "F1_2D", "IoU_2D", "F1_2D_raster", "rho_smooth", "rho_cv")}
    skipped = []

    for i in tqdm(indices, desc="eval", ncols=100):
        sample = val_set[i]
        sample_id = sample.get("id", str(i))
        try:
            M_gt = sample["M"].numpy()
            Z, H, W = M_gt.shape

            prediction = predict_gplan(
                model,
                sample["pc"],
                Z,
                device=device,
                raster_size=(H, W),
                slice_batch_size=args.slice_batch_size,
                raster_meta={
                    "ctr_mm": sample["ctr_mm"],
                    "scale_mm": sample["scale_mm"],
                    "origin_xy_mm": sample["raster_origin_xy_mm"],
                    "px_mm": sample["raster_px_mm"],
                    "raster_H": sample["raster_H"],
                    "raster_W": sample["raster_W"],
                    "scale_to_target": sample["raster_scale_to_target"],
                    "pad_oy": sample["raster_pad_oy"],
                    "pad_ox": sample["raster_pad_ox"],
                },
            )

            iou, f1_raster = slice_iou_f1(prediction.M, M_gt)
            frame = frame_from_sample(sample)

            pred_pts = perimeter_points(prediction.M, frame, R=prediction.R, longest_only=not args.all_contours)
            if pred_pts.size == 0:
                pred_pts = perimeter_points(prediction.M, frame, R=None, longest_only=not args.all_contours)

            stl_path = sample.get("stl_path", "")
            gt_pts = mesh_surface_points(stl_path, args.gt_surface_points) if stl_path else np.zeros((0, 3))
            if gt_pts.size == 0:
                gt_pts = occupancy_points(M_gt, frame)

            pred_pts, _ = align_prediction_to_mesh(
                pred_pts, gt_pts, frame,
                raster_H=int(sample["raster_H"]), raster_W=int(sample["raster_W"]),
            )

            # CD is reported in the normalised frame of the object, F1 in mm.
            center = sample["ctr_mm"].numpy().reshape(3).astype(np.float64)
            scale = max(float(sample["scale_mm"]), 1e-9)
            cd = chamfer_distance((pred_pts - center) / scale, (gt_pts - center) / scale)
            f1_3d = point_f1(pred_pts, gt_pts, tau_mm=args.tau_mm)[2]
            f1_2d = per_slice_point_f1(pred_pts, gt_pts, num_slices=Z, tau_mm=args.tau_mm)

            scores["CD"].append(cd)
            scores["F1_3D"].append(f1_3d)
            scores["F1_2D"].append(f1_2d)
            scores["IoU_2D"].append(iou)
            scores["F1_2D_raster"].append(f1_raster)

            if prediction.Q is not None:
                flow = gplan_flow_metrics(prediction.M, prediction.Q, frame.px_mm)
                scores["rho_smooth"].append(flow["rho_smooth"])
                scores["rho_cv"].append(flow["rho_cv"])

            if not args.quiet:
                tqdm.write(f"  {sample_id}: CD={cd:.4f} F1_3D={f1_3d:.4f} "
                           f"F1_2D={f1_2d:.4f} IoU_2D={iou:.4f}")
        except Exception as exc:  # noqa: BLE001 - a broken mesh must not stop the sweep
            skipped.append(sample_id)
            print(f"  skip {sample_id}: {exc}", file=sys.stderr)

    results = {k: (float(np.nanmean(v)) if v else float("nan")) for k, v in scores.items()}
    results["num_samples"] = len(scores["CD"])
    results["num_skipped"] = len(skipped)
    results["skipped"] = skipped

    print("\n=== results ===")
    print(f"CD      {results['CD']:.4f}")
    print(f"F1_3D   {results['F1_3D']:.4f}")
    print(f"F1_2D   {results['F1_2D']:.4f}")
    print(f"IoU_2D  {results['IoU_2D']:.4f}")
    if scores["rho_smooth"]:
        print(f"rho-smooth {results['rho_smooth']:.4f} | rho-CV {results['rho_cv']:.4f}  (G-plan maps)")
    print(f"objects {results['num_samples']} (skipped {results['num_skipped']})")

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
