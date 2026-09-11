"""Evaluation metrics (paper, Sec. 5.2).

    geometry  CD, F1_3D, F1_2D and the point extraction they run on
    flow      extrusion smoothness and consistency of the generated toolpaths
"""

from .flow import gcode_flow_metrics, gplan_flow_metrics, parse_extrusion_moves
from .geometry import (
    RasterFrame,
    align_isotropic,
    align_prediction_to_mesh,
    chamfer_distance,
    mesh_surface_points,
    occupancy_points,
    per_slice_point_f1,
    perimeter_points,
    point_f1,
    point_inside_validity,
    region_points,
    slice_iou_f1,
)

__all__ = [
    "RasterFrame",
    "align_isotropic",
    "align_prediction_to_mesh",
    "chamfer_distance",
    "mesh_surface_points",
    "occupancy_points",
    "per_slice_point_f1",
    "perimeter_points",
    "point_f1",
    "point_inside_validity",
    "region_points",
    "slice_iou_f1",
    "gcode_flow_metrics",
    "gplan_flow_metrics",
    "parse_extrusion_moves",
]
