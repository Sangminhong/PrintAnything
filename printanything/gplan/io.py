"""Reading and writing G-plan map folders.

A G-plan map is stored as a manifest plus one ``.npy`` per slice and per channel::

    <root>/gplan_manifest.json
    <root>/layers/L000_M.npy      occupancy, uint8   (H, W)
    <root>/layers/L000_R.npy      region,    uint8   (H, W)
    <root>/layers/L000_Q.npy      flow,      float32 (H, W)

The manifest records, for every slice, its physical height and the mapping from
pixels to millimetres (``px_mm`` and ``origin_xy_mm``), which the G-code compiler
needs in order to place toolpaths on the bed.

``ir_manifest.json`` is accepted as an alternative manifest name: it is what the
original research code wrote, and datasets rasterised with it stay usable.
"""

import json
import os
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "MANIFEST_NAMES",
    "find_manifest",
    "save_gplan_folder",
    "load_gplan_stack",
]

MANIFEST_NAMES = ("gplan_manifest.json", "ir_manifest.json")


def find_manifest(folder) -> Optional[str]:
    """Return the manifest path inside ``folder``, or None if there is none."""
    for name in MANIFEST_NAMES:
        path = os.path.join(str(folder), name)
        if os.path.exists(path):
            return path
    return None


def save_gplan_folder(
    out_dir,
    M: np.ndarray,
    R: Optional[np.ndarray] = None,
    Q: Optional[np.ndarray] = None,
    *,
    px_mm: float = 0.20,
    dz_mm: float = 0.30,
    origin_xy_mm: Sequence[float] = (0.0, 0.0),
    z_start_mm: Optional[float] = None,
    sample_id: str = "sample",
    nozzle_diameter: float = 0.4,
    filament_diameter: float = 1.75,
    infill_density: float = 0.20,
    verbose: bool = False,
) -> str:
    """Write a (Z, H, W) G-plan map to ``out_dir`` and return the manifest path.

    ``R`` defaults to an all-gap stub and ``Q`` is omitted when not given, which
    is what the M-only / M+R ablations of Table 4 produce.

    ``z_start_mm`` is the height of the *first* slice; it defaults to one layer
    height, since a first layer at Z0 would drive the nozzle into the bed.
    """
    if z_start_mm is None:
        z_start_mm = dz_mm
    out_dir = str(out_dir)
    Z, H, W = M.shape
    layers_dir = os.path.join(out_dir, "layers")
    os.makedirs(layers_dir, exist_ok=True)

    layers = []
    for zi in range(Z):
        np.save(os.path.join(layers_dir, f"L{zi:03d}_M.npy"), M[zi].astype(np.uint8))
        region = np.zeros((H, W), dtype=np.uint8) if R is None else R[zi].astype(np.uint8)
        np.save(os.path.join(layers_dir, f"L{zi:03d}_R.npy"), region)

        flow_uri = None
        if Q is not None:
            np.save(os.path.join(layers_dir, f"L{zi:03d}_Q.npy"), Q[zi].astype(np.float32))
            flow_uri = f"layers/L{zi:03d}_Q.npy"

        layers.append(
            {
                "id": zi,
                "z_mm": float(z_start_mm + zi * dz_mm),
                "layer_height_mm": float(dz_mm),
                "raster": {
                    "H": int(H),
                    "W": int(W),
                    "px_mm": float(px_mm),
                    "origin_xy_mm": [float(origin_xy_mm[0]), float(origin_xy_mm[1])],
                    "occupancy_uri": f"layers/L{zi:03d}_M.npy",
                    "region_uri": f"layers/L{zi:03d}_R.npy",
                    "flow_uri": flow_uri,
                },
            }
        )

    manifest = {
        "schema_version": "0.3",
        "units": "mm",
        "sample_id": sample_id,
        "global": {"infill_density": float(infill_density), "base_layer_height_mm": float(dz_mm)},
        "printer_profile": {
            "nozzle_diameter": float(nozzle_diameter),
            "filament_diameter": float(filament_diameter),
        },
        "material_profile": {},
        "layers": layers,
    }

    manifest_path = os.path.join(out_dir, MANIFEST_NAMES[0])
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    if verbose:
        print(f"  Saved G-plan map to: {out_dir}")
    return manifest_path


def load_gplan_stack(manifest_path) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Dict]:
    """Load ``(M, R, Q, meta)`` from a G-plan manifest, or from its folder.

    ``R`` and ``Q`` are None when the manifest does not reference them. ``meta``
    carries the raster frame (``origin_xy_mm``, ``px_mm_raw``, ``H_raw``,
    ``W_raw``, ``dz_mm``) plus the identity resize mapping, so it can be fed to
    the printer-frame slice projection directly.
    """
    manifest_path = str(manifest_path)
    if os.path.isdir(manifest_path):
        found = find_manifest(manifest_path)
        if found is None:
            raise FileNotFoundError(f"No G-plan manifest under {manifest_path}")
        manifest_path = found

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    layers = sorted(manifest.get("layers", []), key=lambda x: int(x["id"]))
    if not layers:
        raise ValueError(f"No layers in manifest: {manifest_path}")

    root = os.path.dirname(manifest_path)
    M_list, R_list, Q_list, z_vals = [], [], [], []
    for layer in layers:
        raster = layer["raster"]
        M_list.append(np.load(os.path.join(root, raster["occupancy_uri"])).astype(np.uint8))
        if raster.get("region_uri"):
            R_list.append(np.load(os.path.join(root, raster["region_uri"])).astype(np.uint8))
        if raster.get("flow_uri"):
            Q_list.append(np.load(os.path.join(root, raster["flow_uri"])).astype(np.float32))
        z_vals.append(float(layer.get("z_mm", 0.0)))

    M = np.stack(M_list, axis=0)
    R = np.stack(R_list, axis=0) if len(R_list) == len(M_list) else None
    Q = np.stack(Q_list, axis=0) if len(Q_list) == len(M_list) else None

    first = layers[0]["raster"]
    origin = first.get("origin_xy_mm", [0.0, 0.0])
    if len(z_vals) >= 2:
        dz_mm = float((z_vals[-1] - z_vals[0]) / max(1, len(z_vals) - 1))
    else:
        dz_mm = float(layers[0].get("layer_height_mm", 0.2))

    meta = {
        "origin_xy_mm": (float(origin[0]), float(origin[1])),
        "px_mm_raw": float(first["px_mm"]),
        "H_raw": int(first["H"]),
        "W_raw": int(first["W"]),
        "dz_mm": dz_mm,
        "z_start_mm": z_vals[0] if z_vals else 0.0,
        "scale_raw_to_target": 1.0,
        "pad_oy": 0,
        "pad_ox": 0,
    }
    return M, R, Q, meta
