#!/usr/bin/env python3
"""Generate G-code for the held-out split (Figure 2, end to end).

    python tools/generate_gcode.py \
        --ckpt runs/printanything/checkpoints/ep049.pt \
        --out_dir outputs/val_gcode

For every held-out object this predicts the G-plan map, optionally synthesises an
infill pattern inside the predicted infill regions, writes the G-plan folder, and
compiles it into a ``.gcode`` file. The result is what ``tools/flow_metrics.py``
scores and what a printer can execute.
"""

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import add_data_args, add_gcode_args, dataset_and_split, gcode_config_from_args  # noqa: E402
from printanything.gplan import gplan_to_gcode, save_gplan_folder  # noqa: E402
from printanything.infill import PATTERNS, apply_infill  # noqa: E402
from printanything.inference import predict_gplan  # noqa: E402
from printanything.models import load_gpnet_checkpoint  # noqa: E402
from printanything.utils import resolve_checkpoint, set_seed  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_args(ap)
    add_gcode_args(ap)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", required=True, help="Destination of the .gcode files.")
    ap.add_argument("--gplan_out_dir", default="",
                    help="Keep the intermediate G-plan maps here (default: alongside the G-code).")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--slice_batch_size", type=int, default=8)
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--infill_pattern", default="", choices=("", *PATTERNS),
                    help="Template synthesised inside the predicted infill regions ('' = keep the prediction).")
    ap.add_argument("--infill_scale", type=float, default=1.0)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    model, ckpt = load_gpnet_checkpoint(resolve_checkpoint(args.ckpt), map_location="cpu")
    model = model.to(device).eval()
    print(f"checkpoint: {args.ckpt} (epoch {ckpt.get('epoch', ckpt.get('ep', '?'))})")

    _, _, val_set = dataset_and_split(args, use_R=model.predict_R, use_Q=model.predict_Q)
    count = len(val_set) if not args.max_samples else min(args.max_samples, len(val_set))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gplan_root = Path(args.gplan_out_dir) if args.gplan_out_dir else out_dir / "gplan"
    gplan_root.mkdir(parents=True, exist_ok=True)
    cfg = gcode_config_from_args(args)

    done = failed = 0
    for i in tqdm(range(count), desc="generate", ncols=100):
        sample = val_set[i]
        sample_id = str(sample.get("id", f"sample_{i:05d}")).replace("/", "_").replace(" ", "_")
        try:
            Z, H, W = sample["M"].shape
            prediction = predict_gplan(
                model, sample["pc"], Z,
                device=device,
                raster_size=(H, W),
                slice_batch_size=args.slice_batch_size,
                px_mm=float(sample.get("px_mm_label", 0.2)),
                dz_mm=float(sample.get("dz_mm_label", 0.2)),
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

            M, R = prediction.M, prediction.R
            if args.infill_pattern and R is not None:
                M, R = apply_infill(
                    M, R, pattern=args.infill_pattern, scale=args.infill_scale, device=device
                )

            gplan_dir = gplan_root / sample_id
            save_gplan_folder(
                gplan_dir, M, R, prediction.Q,
                px_mm=prediction.px_mm, dz_mm=prediction.dz_mm, sample_id=sample_id,
            )
            gplan_to_gcode(gplan_dir, out_dir / f"{sample_id}.gcode", cfg, progress=False)
            done += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  fail {sample_id}: {exc}", file=sys.stderr)

    print(f"[done] {done} G-code files in {out_dir} ({failed} failures)")


if __name__ == "__main__":
    main()
