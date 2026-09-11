"""Definition of the Geometric plan (G-plan) map (paper, Sec. 3.1).

A G-plan map is a per-slice triplet

    M  (Z, H, W) uint8    occupancy  -- which pixels carry deposited material
    R  (Z, H, W) uint8    region     -- structural category of each pixel
    Q  (Z, H, W) float32  flow       -- normalised extrusion length per pixel

All maps are rasterised at a fixed ``RASTER_SIZE`` resolution so that they can be
consumed by an image-style encoder-decoder.
"""

RASTER_SIZE = (256, 256)  # (H, W) used for every experiment in the paper

# Structural categories carried by the region map R.
GAP = 0
WALL = 1
INFILL = 2
SUPPORT = 3
SKIRT = 4

NUM_REGION_CLASSES = 5

REGION_ID = {
    "gap": GAP,
    "wall": WALL,
    "infill": INFILL,
    "support": SUPPORT,
    "skirt": SKIRT,
}
REGION_NAME = {v: k for k, v in REGION_ID.items()}

# Regions whose geometry comes from the predicted region map itself and is never
# overwritten by a synthesised infill pattern (Sec. 3.3).
STRUCTURAL_CLASSES = (WALL, SUPPORT, SKIRT)

# Class weights for the weighted cross-entropy of Eq. (6). Walls and supports are
# thin and therefore heavily under-represented relative to gap pixels.
REGION_CLASS_WEIGHTS = (0.1, 3.0, 1.0, 2.0, 0.5)

__all__ = [
    "RASTER_SIZE",
    "GAP",
    "WALL",
    "INFILL",
    "SUPPORT",
    "SKIRT",
    "NUM_REGION_CLASSES",
    "REGION_ID",
    "REGION_NAME",
    "STRUCTURAL_CLASSES",
    "REGION_CLASS_WEIGHTS",
]
