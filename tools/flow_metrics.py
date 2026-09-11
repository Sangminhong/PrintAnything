#!/usr/bin/env python3
"""Extrusion-flow metrics over a folder of G-code (Table 4, Eqs. 12-13).

    python tools/flow_metrics.py --gcode_root outputs/val_gcode --out_json outputs/flow.json

Reports the mean of ``d(rho)-smooth`` and ``rho-CV`` over the files. Both are
lower-is-better and say how steadily material is deposited along the toolpaths;
comparing two runs only makes sense when the same compiler settings were used.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from printanything.evaluation import gcode_flow_metrics  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gcode_root", required=True)
    ap.add_argument("--glob", default="*.gcode")
    ap.add_argument("--out_json", default="")
    ap.add_argument("--per_file", action="store_true", help="Also print every file's numbers.")
    args = ap.parse_args()

    files = sorted(Path(args.gcode_root).glob(args.glob))
    if not files:
        sys.exit(f"No files matching '{args.glob}' under {args.gcode_root}")

    smooth, cv, per_file = [], [], {}
    for path in tqdm(files, desc="flow", ncols=100):
        try:
            metrics = gcode_flow_metrics(str(path))
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {path.name}: {exc}", file=sys.stderr)
            continue
        per_file[path.name] = metrics
        if np.isfinite(metrics["rho_smooth"]):
            smooth.append(metrics["rho_smooth"])
        if np.isfinite(metrics["rho_cv"]):
            cv.append(metrics["rho_cv"])
        if args.per_file:
            tqdm.write(f"  {path.name}: rho-smooth={metrics['rho_smooth']:.5f} "
                       f"rho-CV={metrics['rho_cv']:.5f} moves={metrics['num_moves']}")

    results = {
        "rho_smooth": float(np.mean(smooth)) if smooth else float("nan"),
        "rho_cv": float(np.mean(cv)) if cv else float("nan"),
        "num_files": len(per_file),
        "gcode_root": str(args.gcode_root),
    }
    print(f"\nd(rho)-smooth {results['rho_smooth']:.5f} | rho-CV {results['rho_cv']:.5f} "
          f"over {results['num_files']} files")

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as fh:
            json.dump({**results, "per_file": per_file}, fh, indent=2)
        print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
