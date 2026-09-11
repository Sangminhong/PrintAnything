#!/usr/bin/env python3
"""Rasterise ground-truth G-plan maps from the Slice-100K G-code (Sec. 3.1).

This is the first step of the pipeline: it turns every reference ``.gcode`` file
into the folder of occupancy / region / flow maps that supervises GPNet.

    python tools/prepare_gplan.py \
        --gcode_root data/slice100k/gcode \
        --out_root   data/slice100k_gplan \
        --workers 8

Samples that already have a manifest are skipped, so the command can be resumed.
"""

import argparse
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from printanything.gplan import find_manifest, gcode_to_gplan  # noqa: E402


def convert_one(gcode_path: str, out_dir: str, px_mm, start_after_external_perimeter: bool):
    try:
        path = gcode_to_gplan(
            gcode_path,
            out_dir,
            px_mm=px_mm,
            start_after_external_perimeter=start_after_external_perimeter,
        )
        return gcode_path, path, None
    except Exception as exc:  # noqa: BLE001 - one bad file must not stop the run
        return gcode_path, None, f"{exc}\n{traceback.format_exc(limit=2)}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gcode_root", required=True, help="Directory holding the reference .gcode files.")
    ap.add_argument("--out_root", required=True, help="Where to write one G-plan folder per sample.")
    ap.add_argument("--px_mm", type=float, default=None,
                    help="Pixel size in mm; default is a quarter of the detected line width.")
    ap.add_argument("--glob", default="*.gcode", help="Pattern of the G-code files to convert.")
    ap.add_argument("--limit", type=int, default=0, help="Convert at most N files (0 = all).")
    ap.add_argument("--workers", type=int, default=1, help="Parallel worker processes.")
    ap.add_argument("--overwrite", action="store_true", help="Re-convert samples that already exist.")
    ap.add_argument("--start_after_external_perimeter", action="store_true",
                    help="Start parsing at the first ';TYPE:External perimeter' marker.")
    args = ap.parse_args()

    gcode_files = sorted(Path(args.gcode_root).glob(args.glob))
    if args.limit:
        gcode_files = gcode_files[: args.limit]
    if not gcode_files:
        sys.exit(f"No G-code files matching '{args.glob}' under {args.gcode_root}")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    jobs = []
    skipped = 0
    for gcode in gcode_files:
        out_dir = out_root / gcode.stem
        if not args.overwrite and find_manifest(out_dir) is not None:
            skipped += 1
            continue
        jobs.append((str(gcode), str(out_dir)))

    print(f"{len(gcode_files)} G-code files | {skipped} already converted | {len(jobs)} to do")
    if not jobs:
        return

    done = failed = 0
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(convert_one, g, o, args.px_mm, args.start_after_external_perimeter)
                for g, o in jobs
            ]
            for future in as_completed(futures):
                gcode_path, _, error = future.result()
                done += 1
                if error:
                    failed += 1
                    print(f"[fail] {Path(gcode_path).name}: {error.splitlines()[0]}", file=sys.stderr)
                if done % 25 == 0 or done == len(jobs):
                    print(f"  {done}/{len(jobs)} converted ({failed} failed)")
    else:
        for i, (gcode_path, out_dir) in enumerate(jobs, 1):
            _, _, error = convert_one(gcode_path, out_dir, args.px_mm, args.start_after_external_perimeter)
            if error:
                failed += 1
                print(f"[fail] {Path(gcode_path).name}: {error.splitlines()[0]}", file=sys.stderr)
            if i % 25 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} converted ({failed} failed)")

    print(f"[done] G-plan maps under {out_root} ({failed} failures)")


if __name__ == "__main__":
    main()
