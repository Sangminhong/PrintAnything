"""Small helpers shared by the training, evaluation and tooling entry points."""

import os
import random
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Seed python / numpy / torch so that the train-val split is reproducible.

    The validation split used throughout the paper is obtained from
    ``torch.utils.data.random_split`` with ``--seed 42``; every script that
    reports numbers must therefore use the same seed and ``--val_split``.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path) -> str:
    os.makedirs(path, exist_ok=True)
    return str(path)


def parse_float_list(s: Optional[str]) -> Optional[List[float]]:
    """Parse a comma-separated list of floats (``""`` / ``None`` -> ``None``)."""
    if s is None or str(s).strip() == "":
        return None
    vals = [float(item) for item in str(s).split(",") if item.strip()]
    return vals or None


def parse_int_list(s: Optional[str]) -> Optional[List[int]]:
    vals = parse_float_list(s)
    return None if vals is None else [int(v) for v in vals]


def resolve_checkpoint(ckpt_path) -> Path:
    """Accept both ``<run>/ep120.pt`` and ``<run>/checkpoints/ep120.pt``."""
    ckpt_path = Path(ckpt_path)
    if ckpt_path.exists():
        return ckpt_path
    alt = ckpt_path.parent / "checkpoints" / ckpt_path.name
    if alt.exists():
        return alt
    raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
