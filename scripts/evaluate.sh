#!/usr/bin/env bash
# Evaluate a checkpoint and score the toolpaths it compiles to (Sec. 5.2).
set -euo pipefail

CKPT=${1:?usage: scripts/evaluate.sh <checkpoint> [out_dir]}
OUT_DIR=${2:-outputs/eval}

python tools/evaluate.py      --ckpt "$CKPT" --out_json "$OUT_DIR/metrics.json"
python tools/generate_gcode.py --ckpt "$CKPT" --out_dir "$OUT_DIR/gcode"
python tools/flow_metrics.py  --gcode_root "$OUT_DIR/gcode" --out_json "$OUT_DIR/flow.json"
