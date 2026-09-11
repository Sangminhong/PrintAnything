"""U-Net building blocks used by GPNet (paper, Sec. 3.2)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["group_norm", "DoubleConv", "Down", "Up", "FiLM"]


def group_norm(num_channels: int) -> nn.GroupNorm:
    """GroupNorm with a group count that divides ``num_channels``."""
    for groups in (16, 8, 4, 2):
        if num_channels % groups == 0:
            return nn.GroupNorm(groups, num_channels)
    return nn.GroupNorm(1, num_channels)


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            group_norm(out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            group_norm(out_ch),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    """Upsample, concatenate the skip connection, and fuse.

    ``in_ch`` is the channel count *after* concatenation with the skip.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


class FiLM(nn.Module):
    """Feature-wise linear modulation of the bottleneck, Eq. (1).

        x' = x * (1 + gamma(h)) + beta(h)
    """

    def __init__(self, cond_dim: int, num_channels: int):
        super().__init__()
        self.to_gamma_beta = nn.Linear(cond_dim, num_channels * 2)

    def forward(self, x, cond):
        gamma, beta = self.to_gamma_beta(cond).chunk(2, dim=1)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        return x * (1.0 + gamma) + beta
