#!/usr/bin/env python3
"""Bootstrap the dataset that trains the infill recommender (Sec. 3.3).

For each training object the G-plan map is predicted once, then a number of
random ``(pattern, scale)`` policies are synthesised on top of it and scored with
the strength and cost proxies. Each trial becomes one CSV row:

    sample_id, pattern, scale, layer_idx, strength_proxy, cost_time, ...

    python tools/build_infill_dataset.py \
        --ckpt runs/printanything/checkpoints/ep049.pt \
        --out_csv outputs/infill/trials.csv \
        --num_objects 200 --trials_per_object 12

Everything stays in memory: no G-plan folder is written per trial.
"""

import argparse
import csv
import random
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import add_data_args, dataset_and_split  # noqa: E402
from printanything.infill import PATTERNS, apply_infill, combined_score, compute_proxies  # noqa: E402
from printanything.inference import predict_gplan  # noqa: E402
from printanything.models import load_gpnet_checkpoint  # noqa: E402
from printanything.utils import resolve_checkpoint, set_seed  # noqa: E402

FIELDS = [
    "sample_id", "pattern", "scale", "theta_deg", "layer_idx", "num_slices",
    "strength_proxy", "cost_material", "cost_time", "combined",
    "infill_ratio", "connectivity_score", "direction_balance",
    "occupied_pixels", "infill_pixels", "transition_count",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_args(ap)
    ap.add_argument("--ckpt", required=True, help="Checkpoint used to predict the G-plan maps.")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--slice_batch_size", type=int, default=8)
    ap.add_argument("--num_objects", type=int, default=200, help="Training objects to sample.")
    ap.add_argument("--trials_per_object", type=int, default=12, help="Random policies per object.")
    ap.add_argument("--max_slices", type=int, default=24,
                    help="Evaluate the proxies on at most this many slices per object (0 = all).")
    ap.add_argument("--patterns", default=",".join(PATTERNS))
    ap.add_argument("--scale_min", type=float, default=0.05)
    ap.add_argument("--scale_max", type=float, default=1.5)
    ap.add_argument("--theta_deg", default="0", help="Comma-separated rotations to sample from.")
    args = ap.parse_args()

    set_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device(args.device)

    patterns = [p.strip() for p in args.patterns.split(",") if p.strip()]
    thetas = [float(t) for t in args.theta_deg.split(",") if t.strip()]

    model, _ = load_gpnet_checkpoint(resolve_checkpoint(args.ckpt), map_location="cpu")
    model = model.to(device).eval()

    _, train_set, _ = dataset_and_split(args, use_R=True, use_Q=False)
    object_indices = list(range(len(train_set)))
    rng.shuffle(object_indices)
    object_indices = object_indices[: max(1, args.num_objects)]

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0

    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()

        for i in tqdm(object_indices, desc="trials", ncols=100):
            sample = train_set[i]
            sample_id = str(sample.get("id", f"sample_{i:05d}"))
            try:
                Z, H, W = sample["M"].shape
                prediction = predict_gplan(
                    model, sample["pc"], Z,
                    device=device, raster_size=(H, W), slice_batch_size=args.slice_batch_size,
                )
                if prediction.R is None:
                    continue

                M, R = prediction.M, prediction.R
                if args.max_slices and Z > args.max_slices:
                    step = max(1, Z // args.max_slices)
                    M, R = M[::step][: args.max_slices], R[::step][: args.max_slices]

                for _ in range(args.trials_per_object):
                    pattern = rng.choice(patterns)
                    scale = rng.uniform(args.scale_min, args.scale_max)
                    theta = rng.choice(thetas)

                    M_inf, R_inf = apply_infill(
                        M, R, pattern=pattern, scale=scale, theta_deg=theta, device=device
                    )
                    proxies = compute_proxies(M_inf, R_inf)

                    writer.writerow({
                        "sample_id": sample_id,
                        "pattern": pattern,
                        "scale": round(scale, 5),
                        "theta_deg": theta,
                        "layer_idx": 0,
                        "num_slices": int(M.shape[0]),
                        "combined": combined_score(proxies["strength_proxy"], proxies["cost_time"]),
                        **{k: proxies[k] for k in FIELDS if k in proxies},
                    })
                    rows_written += 1
            except Exception as exc:  # noqa: BLE001
                print(f"  skip {sample_id}: {exc}", file=sys.stderr)

    print(f"[done] {rows_written} trials written to {out_csv}")


if __name__ == "__main__":
    main()
