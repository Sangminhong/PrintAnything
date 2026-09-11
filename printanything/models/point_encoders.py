"""Global point-cloud encoders ``E_global`` (paper, Sec. 3.2).

The encoder maps the input point cloud ``P`` to a single global descriptor
``z_g`` that, together with the slice-height embedding, conditions GPNet through
FiLM.

Two encoders are available:

``ptv3``
    Point Transformer V3 in classification mode, as described in the paper. It
    needs the extra dependencies listed under "Point Transformer V3" in
    ``requirements.txt`` (``spconv``, ``torch-scatter``, ``timm``, ``addict``).
``pointnet``
    A light PointNet-style per-point MLP with max-pooling, with no dependency
    beyond PyTorch. Useful as a fallback and for quick experiments.
"""

import torch
import torch.nn as nn

__all__ = ["PointNetEncoder", "PTv3Encoder", "build_point_encoder", "POINT_ENCODERS"]


class PointNetEncoder(nn.Module):
    """Per-point MLP followed by global max-pooling."""

    def __init__(self, in_dim: int = 3, out_dim: int = 512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, out_dim),
        )

    def forward(self, pc: torch.Tensor) -> torch.Tensor:
        """pc: (B, N, C) or (N, C) -> (B, out_dim)."""
        if pc.dim() == 2:
            pc = pc.unsqueeze(0)
        return self.mlp(pc).max(dim=1)[0]


class PTv3Encoder(nn.Module):
    """Point Transformer V3 backbone with global max-pooling.

    ``third_party/PointTransformerV3`` ships with the release and is put on the
    import path by ``printanything/models/__init__.py``; its own dependencies
    (``spconv``, ``torch-scatter``, ``timm``, ``addict``) must be installed.
    """

    def __init__(
        self,
        in_dim: int = 3,
        out_dim: int = 512,
        grid_size: float = 0.06,
        enc_channels=(32, 64, 128, 256, 512),
        enc_depths=(2, 2, 2, 2, 2),
        enc_num_head=(2, 4, 8, 16, 32),
        patch_size: int = 128,
        enable_flash: bool = False,
    ):
        super().__init__()
        try:
            from PointTransformerV3.model import PointTransformerV3
        except ImportError as exc:  # missing spconv / torch_scatter / timm / addict
            raise ImportError(
                f"The 'ptv3' point encoder needs the Point Transformer V3 dependencies "
                f"({exc}). Install them with:\n"
                f"    pip install addict timm torch-scatter spconv-cu120  # match your CUDA\n"
                f"or use --point_encoder pointnet, which needs nothing beyond PyTorch."
            ) from exc

        self.grid_size = grid_size
        self.backbone = PointTransformerV3(
            in_channels=in_dim,
            enc_channels=enc_channels,
            enc_depths=enc_depths,
            enc_num_head=enc_num_head,
            enc_patch_size=(patch_size,) * len(enc_channels),
            cls_mode=True,  # classification mode: encoder only, no decoder
            enable_flash=enable_flash,  # False avoids the FlashAttention dependency
        )
        self.out_proj = nn.Linear(enc_channels[-1], out_dim)

    def forward(self, pc: torch.Tensor) -> torch.Tensor:
        """pc: (1, N, C) -> (1, out_dim)."""
        if pc.dim() == 2:
            pc = pc.unsqueeze(0)
        B, N, _ = pc.shape
        assert B == 1, "PTv3Encoder processes one point cloud at a time"
        data = {
            "coord": pc[0, :, :3],
            "grid_size": self.grid_size,
            "feat": pc[0],
            "offset": torch.tensor([N], device=pc.device, dtype=torch.long),
        }
        point = self.backbone(data)
        g = point.feat.max(dim=0, keepdim=True)[0]  # (1, enc_channels[-1])
        return self.out_proj(g)


POINT_ENCODERS = {"ptv3": PTv3Encoder, "pointnet": PointNetEncoder}


def build_point_encoder(name: str = "ptv3", in_dim: int = 3, out_dim: int = 512, **kwargs):
    if name not in POINT_ENCODERS:
        raise ValueError(f"Unknown point encoder '{name}'. Choose from {sorted(POINT_ENCODERS)}.")
    return POINT_ENCODERS[name](in_dim=in_dim, out_dim=out_dim, **kwargs)
