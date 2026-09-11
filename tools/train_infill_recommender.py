#!/usr/bin/env python3
"""Fit and evaluate the infill policy recommender (Sec. 3.3, Table 3).

    python tools/train_infill_recommender.py \
        --csv outputs/infill/trials.csv \
        --out_dir outputs/infill/recommender

Objects are split by ``sample_id`` so that no object appears in both train and
test. On the test objects the recommender is compared against the baselines of
Table 3 -- a random policy, and each fixed pattern with a random scale -- on
strength, cost, and the combined score ``1e5 * S / C``.
"""

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from printanything.infill import InfillPolicy, InfillRecommender, combined_score  # noqa: E402


def load_rows(csv_path: str):
    with open(csv_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        for key in ("scale", "layer_idx", "strength_proxy", "cost_time", "cost_material"):
            if key in row and row[key] != "":
                row[key] = float(row[key])
    return rows


def group_split(rows, test_frac: float, seed: int):
    """Split by object id so that the test objects are genuinely unseen."""
    by_object = defaultdict(list)
    for row in rows:
        by_object[row.get("sample_id", "")].append(row)

    keys = sorted(by_object)
    random.Random(seed).shuffle(keys)
    n_test = max(1, int(round(len(keys) * test_frac)))
    test_keys = set(keys[:n_test])

    train = [r for k in keys if k not in test_keys for r in by_object[k]]
    test = {k: by_object[k] for k in test_keys}
    return train, test


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="Trials produced by tools/build_infill_dataset.py.")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--strength_col", default="strength_proxy")
    ap.add_argument("--cost_col", default="cost_time")
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_estimators", type=int, default=300)
    ap.add_argument("--max_depth", type=int, default=14)
    ap.add_argument("--strength_weight", type=float, default=1.0)
    ap.add_argument("--cost_weight", type=float, default=1.0)
    args = ap.parse_args()

    rows = load_rows(args.csv)
    if not rows:
        sys.exit(f"No rows in {args.csv}")
    train_rows, test_objects = group_split(rows, args.test_frac, args.seed)
    print(f"{len(rows)} trials | train {len(train_rows)} rows | test {len(test_objects)} objects")

    recommender = InfillRecommender(
        strength_weight=args.strength_weight,
        cost_weight=args.cost_weight,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        seed=args.seed,
    ).fit(train_rows, strength_col=args.strength_col, cost_col=args.cost_col)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = recommender.save(out_dir / "recommender.pkl")
    print(f"saved {model_path}")

    # --- compare against the Table 3 baselines on the held-out objects --------
    patterns = sorted({str(r["pattern"]) for r in rows})
    rng = random.Random(args.seed)

    def measured(row):
        return float(row[args.strength_col]), float(row[args.cost_col])

    strategies = {"recommender": [], "random_policy": []}
    strategies.update({f"{p}_random_scale": [] for p in patterns})

    for object_rows in test_objects.values():
        candidates = [InfillPolicy(str(r["pattern"]), float(r["scale"])) for r in object_rows]
        best, _ = recommender.recommend(candidates, layer_idx=0)
        picked = next(
            r for r in object_rows
            if str(r["pattern"]) == best.pattern and abs(float(r["scale"]) - best.scale) < 1e-9
        )
        strategies["recommender"].append(measured(picked))
        strategies["random_policy"].append(measured(rng.choice(object_rows)))

        for pattern in patterns:
            same = [r for r in object_rows if str(r["pattern"]) == pattern]
            if same:
                strategies[f"{pattern}_random_scale"].append(measured(rng.choice(same)))

    report = {}
    print(f"\n{'strategy':28s} {'strength':>9s} {'cost':>12s} {'combined':>10s}")
    for name, picks in strategies.items():
        if not picks:
            continue
        strength = float(np.mean([s for s, _ in picks]))
        cost = float(np.mean([c for _, c in picks]))
        combined = combined_score(strength, cost)
        report[name] = {"strength": strength, "cost": cost, "combined": combined, "n": len(picks)}
        print(f"{name:28s} {strength:9.3f} {cost:12.0f} {combined:10.3f}")

    with open(out_dir / "comparison.json", "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {out_dir / 'comparison.json'}")


if __name__ == "__main__":
    main()
