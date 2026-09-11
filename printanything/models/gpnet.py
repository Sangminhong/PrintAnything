"""GPNet: geometric plan map prediction network (paper, Sec. 3.2).

GPNet predicts, for every slice of the object, the three G-plan maps

    M' (occupancy logits)   R' (region logits)   Q' (flow)

from the input point cloud. A global descriptor ``z_g`` of the point cloud and an
embedding of the normalised slice height ``z~`` form the conditioning vector
``h = [z_g ; z_zs]``, which modulates the U-Net bottleneck via FiLM, Eq. (1). The
slice-aligned projection of the point cloud (Sec. 3.2) is the spatial input.
"""

import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..gplan.constants import GAP, NUM_REGION_CLASSES
from ..gplan.slice_projection import (
    num_projection_channels,
    project_slice,
    project_slice_printer_frame,
)
from .point_encoders import build_point_encoder
from .unet import DoubleConv, Down, FiLM, Up

__all__ = ["GPlanDecoder", "GPNet", "load_gpnet_checkpoint"]


class GPlanDecoder(nn.Module):
    """FiLM-conditioned U-Net that decodes one slice of the G-plan map.

    Args:
        global_dim: width of the global point-cloud descriptor ``z_g``.
        hidden_dim: width of the conditioning vector after fusing ``z_g`` and z.
        num_classes_R: number of region classes; 0 disables the region head.
        predict_Q: whether to predict the flow map.
        z_context: MSC context, i.e. neighbouring slabs used per side.
        z_fourier_freqs: if > 0, embed z with that many Fourier bands.
        base_ch: width of the first U-Net stage.
    """

    def __init__(
        self,
        global_dim: int = 512,
        hidden_dim: int = 128,
        num_classes_R: int = NUM_REGION_CLASSES,
        predict_Q: bool = True,
        z_context: int = 1,
        z_fourier_freqs: int = 0,
        base_ch: int = 32,
    ):
        super().__init__()
        self.predict_Q = predict_Q
        self.z_context = int(z_context)
        self.hidden_dim = hidden_dim
        self.z_fourier_freqs = int(z_fourier_freqs)

        # --- conditioning: slice height embedding + global feature ---
        z_in_dim = 1 + 2 * self.z_fourier_freqs
        self.z_embed = nn.Sequential(
            nn.Linear(z_in_dim, 64), nn.ReLU(inplace=True), nn.Linear(64, 128)
        )
        self.feat_combine = nn.Sequential(
            nn.Linear(global_dim + 128, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # --- U-Net over [previous slice occupancy ; projected slice] ---
        self.in_ch = 1 + num_projection_channels(self.z_context)
        b = base_ch
        self.inc = DoubleConv(self.in_ch, b)
        self.down1 = Down(b, b * 2)
        self.down2 = Down(b * 2, b * 4)
        self.down3 = Down(b * 4, b * 8)
        self.down4 = Down(b * 8, b * 16)

        self.film = FiLM(hidden_dim, b * 16)

        self.up1 = Up(b * 16 + b * 8, b * 8)
        self.up2 = Up(b * 8 + b * 4, b * 4)
        self.up3 = Up(b * 4 + b * 2, b * 2)
        self.up4 = Up(b * 2 + b, b)

        self.out_feat = nn.Sequential(
            nn.Conv2d(b, 64, 3, padding=1), nn.GroupNorm(16, 64), nn.SiLU()
        )

        # --- 1x1 prediction heads, Eq. (2) ---
        self.head_M = nn.Conv2d(64, 1, 1)
        self.head_R = (
            nn.Conv2d(64, num_classes_R, 1) if (num_classes_R and num_classes_R > 0) else None
        )
        self.head_Q = nn.Conv2d(64, 1, 1) if predict_Q else None

    def embed_slice_height(self, z_norm: torch.Tensor) -> torch.Tensor:
        """z_norm: (B, 1) in [0, 1] -> (B, 128)."""
        if self.z_fourier_freqs > 0:
            freqs = 2.0 ** torch.arange(
                self.z_fourier_freqs, device=z_norm.device, dtype=z_norm.dtype
            )
            ang = (2.0 * math.pi) * z_norm * freqs[None, :]
            z_feat = torch.cat([z_norm, torch.sin(ang), torch.cos(ang)], dim=1)
        else:
            z_feat = z_norm
        return self.z_embed(z_feat)

    def forward(
        self,
        global_feat: torch.Tensor,
        z_norm: torch.Tensor,
        slice_input: Optional[torch.Tensor] = None,
        prev_slice: Optional[torch.Tensor] = None,
        target_size: Tuple[int, int] = (256, 256),
    ):
        """Decode a batch of slices of the same object.

        Args:
            global_feat: (B, D) global point-cloud feature, broadcast over slices.
            z_norm: (B, 1) normalised slice height in [0, 1].
            slice_input: (B, C, H, W) slice-aligned projection of the point cloud.
            prev_slice: (B, 1, H, W) occupancy of the previous slice, or None.
            target_size: (H, W) of the predicted maps.

        Returns:
            ``(logit_M, logit_R, logit_Q)``; ``logit_R`` / ``logit_Q`` may be None.
        """
        B = global_feat.shape[0]

        cond = self.feat_combine(
            torch.cat([global_feat, self.embed_slice_height(z_norm)], dim=1)
        )  # (B, hidden_dim)

        if prev_slice is None:
            prev_slice = torch.zeros(B, 1, *target_size, device=global_feat.device)
        elif prev_slice.shape[2:] != target_size:
            prev_slice = F.interpolate(
                prev_slice, size=target_size, mode="bilinear", align_corners=False
            )

        if slice_input is None:
            slice_input = torch.zeros(
                B, num_projection_channels(self.z_context), *target_size, device=global_feat.device
            )
        elif slice_input.shape[2:] != target_size:
            slice_input = F.interpolate(slice_input, size=target_size, mode="nearest")

        x = torch.cat([prev_slice, slice_input], dim=1)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.film(self.down4(x4), cond)  # FiLM at the bottleneck, Eq. (1)

        y = self.up1(x5, x4)
        y = self.up2(y, x3)
        y = self.up3(y, x2)
        y = self.up4(y, x1)
        y = self.out_feat(y)

        logit_R = self.head_R(y) if self.head_R is not None else None

        if logit_R is not None:
            # Occupancy is the complement of the "gap" class, which keeps M and R
            # consistent by construction: a gap pixel can never be occupied.
            prob_gap = torch.softmax(logit_R, dim=1)[:, GAP : GAP + 1]
            logit_M = torch.logit((1.0 - prob_gap).clamp(1e-6, 1.0 - 1e-6))
        else:
            logit_M = self.head_M(y)

        logit_Q = self.head_Q(y) if self.head_Q is not None else None
        return logit_M, logit_R, logit_Q


class GPNet(nn.Module):
    """Point cloud -> G-plan map for every slice of the object.

    The point cloud is encoded once; slices are then decoded in chunks of
    ``slice_batch_size`` to bound memory on tall objects.
    """

    def __init__(
        self,
        point_encoder: str = "ptv3",
        global_dim: int = 512,
        num_classes_R: int = NUM_REGION_CLASSES,
        predict_Q: bool = True,
        z_context: int = 1,
        z_fourier_freqs: int = 0,
        encoder_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        self.encoder = build_point_encoder(
            point_encoder, in_dim=3, out_dim=global_dim, **(encoder_kwargs or {})
        )
        self.decoder = GPlanDecoder(
            global_dim=global_dim,
            num_classes_R=num_classes_R,
            predict_Q=predict_Q,
            z_context=z_context,
            z_fourier_freqs=z_fourier_freqs,
        )
        self.point_encoder_name = point_encoder
        self.predict_Q = predict_Q
        self.predict_R = bool(num_classes_R and num_classes_R > 0)

    def forward(
        self,
        pc: torch.Tensor,
        z_values: torch.Tensor,
        target_shapes: Sequence[Tuple[int, int]],
        slice_batch_size: int = 8,
        use_prev_context: bool = True,
        raster_meta: Optional[dict] = None,
    ):
        """Predict the G-plan map of one object.

        Args:
            pc: (1, N, 3) point cloud normalised to [-1, 1].
            z_values: (Z,) slice heights.
            target_shapes: per-slice (H, W); constant within an object.
            slice_batch_size: number of slices decoded per forward pass.
            use_prev_context: feed the previous slice occupancy to the decoder.
            raster_meta: printer-frame raster metadata, as returned by
                :func:`printanything.engine.raster_meta_from_batch`. When given,
                the slice projection is aligned pixel-for-pixel with the GT
                raster; when None it is done in the normalised cube.

        Returns:
            ``(M_slices, R_slices, Q_slices)``, lists of (1, C, H, W) logit maps.
        """
        global_feat = self.encoder(pc)  # (1, D)
        device = global_feat.device

        z_min, z_max = z_values.min(), z_values.max()
        z_range = z_max - z_min
        if z_range < 1e-6:
            z_range = torch.as_tensor(1.0, device=z_values.device, dtype=z_values.dtype)
        z_norm = (z_values - z_min) / z_range

        num_slices = len(z_values)
        dz_norm01 = 1.0 / max(1, num_slices)

        M_slices: List[torch.Tensor] = []
        R_slices: Optional[List[torch.Tensor]] = [] if self.predict_R else None
        Q_slices: Optional[List[torch.Tensor]] = [] if self.predict_Q else None
        prev_M = None

        for start in range(0, num_slices, slice_batch_size):
            end = min(start + slice_batch_size, num_slices)
            chunk = end - start

            target_size = target_shapes[start]
            Ht, Wt = target_size
            assert all(
                target_shapes[i] == target_size for i in range(start, end)
            ), "all slices of one object must share the same (H, W)"

            z_batch = z_norm[start:end].unsqueeze(1).to(device)
            global_feat_batch = global_feat.expand(chunk, -1)

            projections = []
            for i in range(chunk):
                z_scalar = float(z_norm[start + i].item())
                if raster_meta is not None:
                    projections.append(
                        project_slice_printer_frame(
                            pc[0],
                            z_center_norm01=z_scalar,
                            dz_norm01=dz_norm01,
                            out_H=Ht,
                            out_W=Wt,
                            device=device,
                            z_context=self.decoder.z_context,
                            **raster_meta,
                        )
                    )
                else:
                    projections.append(
                        project_slice(
                            pc[0],
                            z_scalar,
                            dz_norm01,
                            Ht,
                            Wt,
                            device=device,
                            z_context=self.decoder.z_context,
                        )
                    )
            slice_input = torch.cat(projections, dim=0)  # (chunk, C, H, W)

            # The same previous-slice context is shared by the whole chunk. This
            # relaxes the strict autoregressive dependency inside a chunk in
            # exchange for a large speed-up.
            prev_batch = prev_M.expand(chunk, -1, -1, -1) if (use_prev_context and prev_M is not None) else None

            logit_M, logit_R, logit_Q = self.decoder(
                global_feat_batch,
                z_batch,
                slice_input=slice_input,
                prev_slice=prev_batch,
                target_size=target_size,
            )

            for i in range(chunk):
                M_slices.append(logit_M[i : i + 1])
                if self.predict_R and logit_R is not None:
                    R_slices.append(logit_R[i : i + 1])
                if self.predict_Q and logit_Q is not None:
                    Q_slices.append(logit_Q[i : i + 1])

            if use_prev_context:
                prev_M = (logit_M[-1:] > 0).float().detach()

            if device.type == "cuda":
                torch.cuda.empty_cache()

        return M_slices, R_slices, Q_slices


def _upgrade_legacy_state_dict(state: dict) -> dict:
    """Map checkpoints trained with the original research code onto ``GPNet``.

    Only the FiLM projection was renamed (``decoder.film.{weight,bias}`` ->
    ``decoder.film.to_gamma_beta.{weight,bias}``); every other parameter keeps
    its original name.
    """
    out = {}
    for k, v in state.items():
        if k.startswith("decoder.film.") and not k.startswith("decoder.film.to_gamma_beta."):
            k = k.replace("decoder.film.", "decoder.film.to_gamma_beta.")
        out[k] = v
    return out


def load_gpnet_checkpoint(ckpt_path, map_location="cpu", **model_kwargs):
    """Instantiate a :class:`GPNet` and load ``ckpt_path`` into it.

    The point encoder is inferred from the checkpoint unless it is given
    explicitly, so that checkpoints trained with either encoder just work.

    Returns:
        ``(model, checkpoint_dict)``.
    """
    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    state = _upgrade_legacy_state_dict(ckpt.get("model", ckpt))

    if "point_encoder" not in model_kwargs:
        model_kwargs["point_encoder"] = (
            "pointnet" if any(k.startswith("encoder.mlp.") for k in state) else "ptv3"
        )
    saved_args = ckpt.get("args", {}) or {}
    model_kwargs.setdefault("z_context", saved_args.get("z_context", saved_args.get("hint_z_context", 1)))
    model_kwargs.setdefault("z_fourier_freqs", saved_args.get("z_fourier_freqs", 0))

    model = GPNet(**model_kwargs)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[load_gpnet_checkpoint] missing={list(missing)} unexpected={list(unexpected)}")
    return model, ckpt
