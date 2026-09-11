#!/usr/bin/env python3
"""Train GPNet on Slice-100K (paper, Sec. 4).

    python tools/train.py \
        --stl_root data/slice100k/stls \
        --gplan_root data/slice100k_gplan \
        --out_root runs/printanything

Defaults follow the paper: 30k input points, 256x256 maps, AdamW at 2e-4 with
weight decay 1e-4, mixed precision, 50 epochs. One object is processed per step
(objects differ in slice count and raster size), and a random subset of its
slices is supervised; see ``--slices_per_step``.

Ablations:
    --z_context 0        no multi-slice conditioning (Table 2)
    --no_region          occupancy only, M                (Table 4, row 1)
    --no_flow            occupancy + region, M + R        (Table 4, row 2)
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import add_data_args, dataset_and_split  # noqa: E402
from printanything.engine import (  # noqa: E402
    AugmentConfig,
    LossConfig,
    collate_objects,
    train_one_epoch,
    validate,
)
from printanything.gplan.constants import NUM_REGION_CLASSES, REGION_CLASS_WEIGHTS  # noqa: E402
from printanything.models import GPNet  # noqa: E402
from printanything.utils import ensure_dir, parse_float_list, parse_int_list, set_seed  # noqa: E402


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_args(ap)

    opt = ap.add_argument_group("optimisation")
    opt.add_argument("--out_root", required=True, help="Run directory for checkpoints and logs.")
    opt.add_argument("--epochs", type=int, default=50)
    opt.add_argument("--lr", type=float, default=2e-4)
    opt.add_argument("--weight_decay", type=float, default=1e-4)
    opt.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    opt.add_argument("--num_workers", type=int, default=4)
    opt.add_argument("--no_amp", action="store_true", help="Disable mixed precision.")
    opt.add_argument("--slices_per_step", type=int, default=22,
                     help="Slices supervised per object per step (first, last and random middle ones).")
    opt.add_argument("--slice_batch_size", type=int, default=4,
                     help="Slices decoded per forward pass; lower it if you run out of memory.")
    opt.add_argument("--val_slice_batch_size", type=int, default=8)
    opt.add_argument("--save_every", type=int, default=10, help="Checkpoint every N epochs.")
    opt.add_argument("--resume", default="", help="Checkpoint to resume from.")

    model = ap.add_argument_group("model")
    model.add_argument("--point_encoder", default="ptv3", choices=("ptv3", "pointnet"),
                       help="Global point-cloud encoder (paper: ptv3).")
    model.add_argument("--global_dim", type=int, default=512)
    model.add_argument("--z_context", type=int, default=1,
                       help="Multi-slice conditioning context; 0 disables MSC (Table 2).")
    model.add_argument("--z_fourier_freqs", type=int, default=0,
                       help="Fourier bands for the slice-height embedding (0 = raw z).")
    model.add_argument("--no_region", action="store_true", help="Do not predict the region map R.")
    model.add_argument("--no_flow", action="store_true", help="Do not predict the flow map Q.")

    loss = ap.add_argument_group("loss")
    loss.add_argument("--lambda_M", type=float, default=1.0)
    loss.add_argument("--lambda_R", type=float, default=1.0)
    loss.add_argument("--lambda_Q", type=float, default=1.0)
    loss.add_argument("--flow_loss", default="huber", choices=("huber", "l1_sum"),
                      help="Masked Huber of Eq. (7), or the unnormalised L1 of the original runs.")

    aug = ap.add_argument_group("scan-artifact augmentation (Sec. 5.6)")
    aug.add_argument("--noise_every", type=int, default=0,
                     help="Apply Gaussian jitter every N steps (0 = never).")
    aug.add_argument("--noise_sigma", type=float, default=0.01, help="Jitter sigma in normalised coords.")
    aug.add_argument("--noise_sigmas", default="", help="Comma-separated sigmas sampled per firing.")
    aug.add_argument("--hole_every", type=int, default=0,
                     help="Apply structured hole dropout every N steps (0 = never).")
    aug.add_argument("--hole_radius", type=float, default=0.12)
    aug.add_argument("--hole_frac", type=float, default=0.06, help="Target fraction of points removed.")
    aug.add_argument("--hole_fracs", default="", help="Comma-separated drop fractions sampled per firing.")

    scan = ap.add_argument_group("simulated depth scans")
    scan.add_argument("--simulated_scans", action="store_true",
                      help="Train on simulated multi-view depth scans instead of uniform mesh samples.")
    scan.add_argument("--scan_views", default="1,2,4,8")
    scan.add_argument("--scan_image_size", type=int, default=256)
    scan.add_argument("--scan_fov_deg", type=float, default=60.0)
    scan.add_argument("--scan_clean_prob", type=float, default=0.0,
                      help="Probability of falling back to the clean uniform point cloud.")
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    out_root = ensure_dir(args.out_root)

    predict_R = not args.no_region
    predict_Q = not args.no_flow

    scan_kwargs = {}
    if args.simulated_scans:
        scan_kwargs = dict(
            simulated_scans=True,
            scan_views=parse_int_list(args.scan_views) or [1, 2, 4, 8],
            scan_image_size=args.scan_image_size,
            scan_fov_deg=args.scan_fov_deg,
            clean_prob=args.scan_clean_prob,
        )

    print("Loading dataset ...")
    _, train_set, val_set = dataset_and_split(args, use_R=predict_R, use_Q=predict_Q, **scan_kwargs)
    print(f"train {len(train_set)} | val {len(val_set)}")

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_objects)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_objects)

    model = GPNet(
        point_encoder=args.point_encoder,
        global_dim=args.global_dim,
        num_classes_R=(NUM_REGION_CLASSES if predict_R else 0),
        predict_Q=predict_Q,
        z_context=args.z_context,
        z_fourier_freqs=args.z_fourier_freqs,
    ).to(device)
    print(f"GPNet | encoder={args.point_encoder} | "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    loss_cfg = LossConfig(
        lambda_M=args.lambda_M,
        lambda_R=args.lambda_R,
        lambda_Q=args.lambda_Q,
        flow_loss=args.flow_loss,
        region_class_weights=list(REGION_CLASS_WEIGHTS),
    )
    augment = AugmentConfig(
        noise_every=args.noise_every,
        noise_sigma=args.noise_sigma,
        noise_sigmas=parse_float_list(args.noise_sigmas),
        hole_every=args.hole_every,
        hole_radius=args.hole_radius,
        hole_target_frac=args.hole_frac,
        hole_fracs=parse_float_list(args.hole_fracs),
    )

    with open(os.path.join(out_root, "args.json"), "w") as fh:
        json.dump(vars(args), fh, indent=2)
    log_path = os.path.join(out_root, "metrics.csv")
    if start_epoch == 0:
        with open(log_path, "w") as fh:
            fh.write("epoch,train_loss,train_IoU_M,val_IoU_M,val_mIoU_R,val_L1_Q\n")

    ckpt_dir = ensure_dir(os.path.join(out_root, "checkpoints"))
    for epoch in range(start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model, train_loader, optimizer, device,
            loss_cfg=loss_cfg,
            augment=augment,
            slices_per_step=args.slices_per_step,
            slice_batch_size=args.slice_batch_size,
            amp=not args.no_amp,
        )
        val_stats = validate(model, val_loader, device, slice_batch_size=args.val_slice_batch_size)

        print(f"[ep {epoch:03d}] loss {train_stats['loss']:.3f} | "
              f"val IoU(M) {val_stats['IoU_M']:.3f} "
              f"mIoU(R) {val_stats['mIoU_R']:.3f} L1(Q) {val_stats['L1_Q']:.3f}")
        with open(log_path, "a") as fh:
            fh.write(f"{epoch},{train_stats['loss']:.6f},{train_stats['iou_M']:.6f},"
                     f"{val_stats['IoU_M']:.6f},{val_stats['mIoU_R']:.6f},{val_stats['L1_Q']:.6f}\n")

        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            path = os.path.join(ckpt_dir, f"ep{epoch:03d}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "args": vars(args),
                },
                path,
            )
            print(f"  saved {path}")

    print("Training complete.")


if __name__ == "__main__":
    main()
