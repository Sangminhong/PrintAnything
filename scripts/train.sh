#!/usr/bin/env bash
# Train GPNet with the settings of the paper (Sec. 4).
set -euo pipefail

STL_ROOT=${STL_ROOT:-data/slice100k/stls}
GPLAN_ROOT=${GPLAN_ROOT:-data/slice100k_gplan}
OUT_ROOT=${OUT_ROOT:-runs/printanything}

python tools/train.py \
    --stl_root   "$STL_ROOT" \
    --gplan_root "$GPLAN_ROOT" \
    --out_root   "$OUT_ROOT" \
    --epochs 50 --lr 2e-4 --weight_decay 1e-4 \
    --num_points 30000 --target_h 256 --target_w 256 \
    --point_encoder "${POINT_ENCODER:-ptv3}" \
    "$@"
