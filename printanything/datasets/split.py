"""The single definition of the train / test split.

The paper partitions Slice-100K 9:1 and reports the mean over the held-out
tenth. Training, evaluation, G-code generation and every baseline therefore have
to rebuild *exactly* the same split: the helpers below are the only place where
it is defined, so a stray argument cannot silently change which objects are
scored.

``--val_split``, ``--seed`` and ``--max_objects`` together determine which
objects are held out; all three have to match for two numbers to be comparable.
"""

from typing import Optional, Tuple

import torch
from torch.utils.data import Subset

from .gplan_dataset import GPlanDataset
from .scan_simulation import SimulatedScanGPlanDataset

__all__ = ["build_dataset", "train_val_split"]


def build_dataset(
    stl_root: str,
    gplan_root: str,
    *,
    num_points: int = 30000,
    use_R: bool = True,
    use_Q: bool = False,
    target_h: Optional[int] = 256,
    target_w: Optional[int] = 256,
    q_cache_root: Optional[str] = None,
    max_objects: Optional[int] = None,
    simulated_scans: bool = False,
    cache_index: bool = True,
    cache_path: Optional[str] = None,
    **scan_kwargs,
) -> GPlanDataset:
    """Instantiate the dataset the same way for every entry point."""
    cls = SimulatedScanGPlanDataset if simulated_scans else GPlanDataset
    return cls(
        stl_root=stl_root,
        gplan_root=gplan_root,
        num_points=num_points,
        use_R=use_R,
        use_Q=use_Q,
        normalize_pc=True,
        cache_index=cache_index,
        cache_path=cache_path,
        target_h=target_h,
        target_w=target_w,
        q_cache_root=q_cache_root,
        max_objects=max_objects,
        **(scan_kwargs if simulated_scans else {}),
    )


def train_val_split(
    dataset, val_split: float = 0.1, seed: int = 42
) -> Tuple[Subset, Subset]:
    """Split 9:1 with a fixed generator seed (paper, Sec. 5.1)."""
    n_total = len(dataset)
    n_val = int(n_total * val_split)
    return torch.utils.data.random_split(
        dataset,
        [n_total - n_val, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
