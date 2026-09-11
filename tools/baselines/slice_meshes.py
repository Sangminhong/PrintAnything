#!/usr/bin/env python3
"""Slice reconstructed meshes into G-code with PrusaSlicer (Table 1).

The mesh-based baselines are "point cloud -> mesh -> slicer -> G-code", and this
is the slicer step. It shells out to the PrusaSlicer CLI:

    python tools/baselines/slice_meshes.py \
        --mesh_dir outputs/poisson/meshes \
        --out_dir  outputs/poisson/gcode \
        --prusa_slicer /usr/bin/prusa-slicer \
        --config configs/prusa_slice100k.ini

Use the same profile the reference G-code of Slice-100K was produced with,
otherwise the comparison mixes geometry errors with profile differences. Newer
PrusaSlicer builds emit binary ``.bgcode``; pass ``--bgcode_bin`` (libbgcode's
``bgcode`` tool) and the files are converted to plain ``.gcode`` afterwards.

A mesh the slicer rejects (non-manifold, inverted faces, empty after repair) is
reported and skipped -- that failure mode is itself part of what Sec. 1 argues
against mesh-based pipelines, so the count is worth keeping.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm


def run(cmd, timeout: int):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mesh_dir", required=True, help="Directory of reconstructed meshes.")
    ap.add_argument("--out_dir", required=True, help="Where to write the .gcode files.")
    ap.add_argument("--glob", default="*.ply", help="Mesh pattern (*.ply, *.obj, *.stl).")
    ap.add_argument("--prusa_slicer", default="prusa-slicer", help="PrusaSlicer executable.")
    ap.add_argument("--config", default="", help="PrusaSlicer .ini profile (strongly recommended).")
    ap.add_argument("--extra_args", default="",
                    help="Extra CLI arguments, e.g. '--fill-density 20 --layer-height 0.2'.")
    ap.add_argument("--bgcode_bin", default="", help="libbgcode 'bgcode' tool, to convert .bgcode output.")
    ap.add_argument("--timeout", type=int, default=600, help="Per-mesh timeout in seconds.")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    slicer = shutil.which(args.prusa_slicer) or args.prusa_slicer
    if not Path(slicer).exists() and shutil.which(args.prusa_slicer) is None:
        sys.exit(f"PrusaSlicer not found: {args.prusa_slicer}")

    meshes = sorted(Path(args.mesh_dir).glob(args.glob))
    if not meshes:
        sys.exit(f"No meshes matching '{args.glob}' under {args.mesh_dir}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    extra = args.extra_args.split() if args.extra_args else []

    done = failed = skipped = 0
    for mesh in tqdm(meshes, desc="slice", ncols=100):
        out_gcode = out_dir / f"{mesh.stem}.gcode"
        if out_gcode.exists() and not args.overwrite:
            skipped += 1
            continue

        cmd = [slicer, "--export-gcode", "--output", str(out_gcode)]
        if args.config:
            cmd += ["--load", args.config]
        cmd += extra + [str(mesh)]

        try:
            result = run(cmd, args.timeout)
        except subprocess.TimeoutExpired:
            failed += 1
            print(f"  timeout {mesh.name}", file=sys.stderr)
            continue

        produced = out_gcode if out_gcode.exists() else None
        if produced is None:
            # Some builds append .bgcode instead of honouring --output verbatim.
            candidates = list(out_dir.glob(f"{mesh.stem}*.bgcode"))
            if candidates and args.bgcode_bin:
                converted = run([args.bgcode_bin, str(candidates[0]), str(out_gcode)], args.timeout)
                if out_gcode.exists():
                    produced = out_gcode
                elif converted.returncode != 0:
                    print(f"  bgcode conversion failed for {mesh.name}: "
                          f"{converted.stderr.strip()[:200]}", file=sys.stderr)

        if produced is None:
            failed += 1
            message = (result.stderr or result.stdout or "").strip().splitlines()
            print(f"  fail {mesh.name}: {message[-1] if message else 'no output produced'}",
                  file=sys.stderr)
            continue
        done += 1

    print(f"[done] {done} sliced | {failed} failed | {skipped} already present -> {out_dir}")


if __name__ == "__main__":
    main()
