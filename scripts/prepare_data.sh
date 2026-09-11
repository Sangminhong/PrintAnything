#!/usr/bin/env bash
# Rasterise the ground-truth G-plan maps from the Slice-100K G-code (Sec. 3.1).
set -euo pipefail

GCODE_ROOT=${GCODE_ROOT:-data/slice100k/gcode}
GPLAN_ROOT=${GPLAN_ROOT:-data/slice100k_gplan}
WORKERS=${WORKERS:-8}

python tools/prepare_gplan.py \
    --gcode_root "$GCODE_ROOT" \
    --out_root   "$GPLAN_ROOT" \
    --workers    "$WORKERS"
