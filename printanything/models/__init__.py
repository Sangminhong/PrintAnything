"""Models: global point-cloud encoders and the GPNet slice decoder."""

import sys
from pathlib import Path

# ``third_party/PointTransformerV3`` ships with the release; expose it on the
# import path so ``from PointTransformerV3.model import ...`` works from a plain
# checkout without an extra install step.
_THIRD_PARTY = Path(__file__).resolve().parents[2] / "third_party"
if _THIRD_PARTY.is_dir() and str(_THIRD_PARTY) not in sys.path:
    sys.path.insert(0, str(_THIRD_PARTY))

from .gpnet import GPNet, GPlanDecoder, load_gpnet_checkpoint  # noqa: E402
from .point_encoders import (  # noqa: E402
    POINT_ENCODERS,
    PointNetEncoder,
    PTv3Encoder,
    build_point_encoder,
)

__all__ = [
    "GPNet",
    "GPlanDecoder",
    "load_gpnet_checkpoint",
    "POINT_ENCODERS",
    "PointNetEncoder",
    "PTv3Encoder",
    "build_point_encoder",
]
