"""The Geometric plan (G-plan) map: definition, construction, and compilation.

    gcode_to_gplan     sliced G-code  -> ground-truth G-plan map   (Sec. 3.1)
    slice_projection   point cloud    -> slice-aligned 2D input    (Sec. 3.2)
    gplan_to_gcode     G-plan map     -> executable G-code         (Sec. 3.4)
    io                 read/write G-plan folders
"""

from .constants import (
    GAP,
    INFILL,
    NUM_REGION_CLASSES,
    RASTER_SIZE,
    REGION_CLASS_WEIGHTS,
    REGION_ID,
    REGION_NAME,
    SKIRT,
    STRUCTURAL_CLASSES,
    SUPPORT,
    WALL,
)
from .gcode_to_gplan import gcode_to_gplan
from .gplan_to_gcode import GCodeConfig, gplan_to_gcode
from .io import find_manifest, load_gplan_stack, save_gplan_folder
from .slice_projection import (
    num_projection_channels,
    project_slice,
    project_slice_printer_frame,
)

__all__ = [
    "GAP",
    "WALL",
    "INFILL",
    "SUPPORT",
    "SKIRT",
    "NUM_REGION_CLASSES",
    "RASTER_SIZE",
    "REGION_CLASS_WEIGHTS",
    "REGION_ID",
    "REGION_NAME",
    "STRUCTURAL_CLASSES",
    "gcode_to_gplan",
    "GCodeConfig",
    "gplan_to_gcode",
    "find_manifest",
    "load_gplan_stack",
    "save_gplan_folder",
    "num_projection_channels",
    "project_slice",
    "project_slice_printer_frame",
]
