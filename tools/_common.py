"""Argument groups shared by the command line tools.

Keeping the dataset and split flags in one place is what guarantees that
training, evaluation, G-code generation and the baselines all see the same
held-out objects.
"""

import argparse
import sys
from pathlib import Path

# Allow running the tools straight from a checkout, without installing.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from printanything.datasets import build_dataset, train_val_split  # noqa: E402

__all__ = ["add_data_args", "dataset_and_split", "add_gcode_args", "gcode_config_from_args"]


def add_data_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group("dataset")
    group.add_argument("--stl_root", default="data/slice100k/stls",
                       help="Root of the Slice-100K STL meshes.")
    group.add_argument("--gplan_root", default="data/slice100k_gplan",
                       help="Root of the ground-truth G-plan maps (see tools/prepare_gplan.py).")
    group.add_argument("--num_points", type=int, default=30000,
                       help="Surface points sampled per object.")
    group.add_argument("--target_h", type=int, default=256, help="Raster height (0 = keep original).")
    group.add_argument("--target_w", type=int, default=256, help="Raster width (0 = keep original).")
    group.add_argument("--q_cache_root", default="",
                       help="Optional cache directory for normalised flow maps.")
    group.add_argument("--index_cache", default="",
                       help="Optional path of the dataset index cache (.json).")
    group.add_argument("--max_objects", type=int, default=0,
                       help="Use only the first N indexed objects (0 = all). "
                            "The paper's runs used 1000; see docs/REPRODUCE.md.")
    group.add_argument("--val_split", type=float, default=0.1, help="Held-out fraction (paper: 0.1).")
    group.add_argument("--seed", type=int, default=42, help="Split / training seed (paper: 42).")
    return parser


def dataset_and_split(args, *, use_R: bool = True, use_Q: bool = False, **extra):
    """Build the dataset and reproduce the 9:1 split exactly."""
    dataset = build_dataset(
        stl_root=args.stl_root,
        gplan_root=args.gplan_root,
        num_points=args.num_points,
        use_R=use_R,
        use_Q=use_Q,
        target_h=args.target_h or None,
        target_w=args.target_w or None,
        q_cache_root=args.q_cache_root or None,
        cache_path=args.index_cache or None,
        max_objects=args.max_objects or None,
        **extra,
    )
    train_set, val_set = train_val_split(dataset, val_split=args.val_split, seed=args.seed)
    return dataset, train_set, val_set


def add_gcode_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group("G-code compiler")
    group.add_argument("--perimeter_speed", type=float, default=40.0, help="mm/s")
    group.add_argument("--infill_speed", type=float, default=60.0, help="mm/s")
    group.add_argument("--travel_speed", type=float, default=150.0, help="mm/s")
    group.add_argument("--line_width", type=float, default=0.45, help="mm")
    group.add_argument("--walls", type=int, default=3, help="Number of perimeter passes.")
    group.add_argument("--infill_density", type=float, default=1.0, help="0..1")
    group.add_argument("--infill_overlap", type=float, default=0.12, help="Wall/infill overlap fraction.")
    group.add_argument("--perimeter_eps_px", type=float, default=0.5,
                       help="Contour simplification tolerance, in pixels.")
    group.add_argument("--max_volumetric_flow", type=float, default=None, help="mm^3/s")
    group.add_argument("--no_flow", action="store_true",
                       help="Ignore the flow map Q when computing extrusion.")
    group.add_argument("--flow_scale", type=float, default=1.0, help="Multiplier applied to Q.")
    group.add_argument("--bed_center_xy_mm", default="",
                       help="'x,y' of the bed centre; recentres the object for real printing.")
    return parser


def gcode_config_from_args(args):
    from printanything.gplan import GCodeConfig

    bed_center = None
    if getattr(args, "bed_center_xy_mm", ""):
        x, y = (float(v) for v in args.bed_center_xy_mm.split(","))
        bed_center = (x, y)

    return GCodeConfig(
        perimeter_speed=args.perimeter_speed,
        infill_speed=args.infill_speed,
        travel_speed=args.travel_speed,
        line_width=args.line_width,
        walls=args.walls,
        infill_density=args.infill_density,
        infill_overlap=args.infill_overlap,
        perimeter_eps_px=args.perimeter_eps_px,
        max_volumetric_flow=args.max_volumetric_flow,
        use_flow=not args.no_flow,
        flow_scale=args.flow_scale,
        bed_center_xy_mm=bed_center,
    )
