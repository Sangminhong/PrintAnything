"""Infill pattern generation (paper, Sec. 3.3).

    patterns     the template dictionary and how a template is inserted
    proxies      strength / cost objectives of an infilled G-plan map
    recommender  surrogate model that picks the (pattern, scale) policy
"""

from .patterns import (
    MONOTONIC,
    PATTERNS,
    apply_infill,
    default_pattern_dir,
    load_template,
    warp_template_periodic,
)
from .proxies import combined_score, compute_proxies, largest_component_ratio
from .recommender import InfillPolicy, InfillRecommender, pareto_front

__all__ = [
    "MONOTONIC",
    "PATTERNS",
    "apply_infill",
    "default_pattern_dir",
    "load_template",
    "warp_template_periodic",
    "combined_score",
    "compute_proxies",
    "largest_component_ratio",
    "InfillPolicy",
    "InfillRecommender",
    "pareto_front",
]
