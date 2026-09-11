"""Ground-truth G-plan map generation (paper, Sec. 3.1).

Given a sliced G-code file, every extrusion move is parsed into a segment with
its structural category and deposited volume, and the segments of each slice are
rasterised into the three G-plan maps:

  * the region map ``R`` rasterises the structural categories declared by the
    G-code (``;TYPE:`` comments), resolving overlaps by priority so that a wall
    is never overwritten by the infill that abuts it;
  * the occupancy map ``M`` is the union of all printed regions;
  * the flow map ``Q`` holds the extruded volume rate of the segment covering
    each pixel, averaged where several segments overlap.

Layers are taken from the slicer's ``;LAYER_CHANGE`` / ``;Z:`` markers when they
exist. Otherwise a new layer starts at the first extrusion that follows a Z
increase, which keeps Z-hop travel moves from splitting a layer -- this is what
makes the same code usable on G-code from other slicers, and it is how the
baselines' G-code is rasterised for evaluation.

The pixel size defaults to a quarter of the extrusion width, and all layers of an
object share one raster frame so that the stack is a well-formed (Z, H, W) array.
"""

import json
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .constants import GAP, INFILL, SKIRT, SUPPORT, WALL
from .constants import REGION_ID as _REGION_ID
from .io import MANIFEST_NAMES


def filament_area(diam_mm: float) -> float:
    r = diam_mm * 0.5
    return math.pi * r * r


def distance(x0, y0, x1, y1):
    return math.hypot(x1 - x0, y1 - y0)


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


# "perimeter" is what slicers call the structural category the paper calls WALL.
REGION_ID = dict(_REGION_ID, perimeter=WALL)

# Overlapping segments: the higher priority wins the pixel.
REGION_PRIORITY = {GAP: 0, SKIRT: 1, INFILL: 2, SUPPORT: 3, WALL: 4}


def map_type_to_region(type_str: Optional[str]) -> int:
    if not type_str:
        return REGION_ID["infill"]
    t = type_str.lower()
    if "skirt" in t or "brim" in t:
        return REGION_ID["skirt"]
    if "support" in t:
        return REGION_ID["support"]
    if "perimeter" in t:
        return REGION_ID["perimeter"]
    if "infill" in t:
        return REGION_ID["infill"]
    if "wall" in t:
        # common in some slicers: WALL-INNER/OUTER
        return REGION_ID["perimeter"]
    return REGION_ID["infill"]


@dataclass
class Segment:
    x0: float
    y0: float
    x1: float
    y1: float
    z: float
    L: float
    v_mm_s: float
    dE_mm: float
    width_mm: float
    height_mm: float
    region_id: int


@dataclass
class LayerRaster:
    z_mm: float
    height_mm: float
    H: int
    W: int
    px_mm: float
    origin_xy_mm: Tuple[float, float]
    M: np.ndarray
    R: np.ndarray
    Q: np.ndarray


class GCodeParser:
    """
    Minimal single-extruder parser that accumulates extrusion segments and
    groups them into layers.
    """

    def __init__(self, start_after_external_perimeter: bool = False):
        self.abs_xyz = True
        self.abs_e = True
        self.units_mm = True
        self.start_after_external_perimeter = start_after_external_perimeter

        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.e = 0.0
        self.f = 0.0

        self.active_type = None
        self.active_width = None
        self.active_height = None

        self.layer_index = -1
        self.layer_z = None
        self.layer_height = None

        self.header_meta = {}

        # When we detect Z increases without explicit layer-change tags, we
        # defer starting a new layer until we see the next extrusion.
        self._pending_layer_z: Optional[float] = None

    def parse_file(self, path: str) -> Tuple[Dict[int, Dict], Dict]:
        layers: Dict[int, Dict] = {}
        started_geometry = (not self.start_after_external_perimeter)

        def start_new_layer(z_val: float, height_val: Optional[float] = None):
            self.layer_index += 1
            self.layer_z = z_val
            if height_val is not None:
                self.layer_height = height_val
            if self.layer_index not in layers:
                layers[self.layer_index] = {"z": z_val, "height": self.layer_height, "segs": []}

        re_float = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
        rx_type = re.compile(r";\s*TYPE\s*:\s*(.+)", re.I)
        rx_width = re.compile(r";\s*WIDTH\s*:\s*(" + re_float + r")", re.I)
        rx_height = re.compile(r";\s*HEIGHT\s*:\s*(" + re_float + r")", re.I)
        rx_layer_change = re.compile(r";\s*LAYER_CHANGE", re.I)
        rx_z_tag = re.compile(r";\s*Z\s*:\s*(" + re_float + r")", re.I)
        rx_meta_kv = re.compile(r";\s*([A-Za-z_ ]+)\s*=\s*(.*)")

        eps_z = 1e-6

        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            prev_e_for_abs = 0.0
            have_prev_e = False
            pending_new_layer = False  # slicer-style tags

            for raw in fh:
                line = raw.strip()

                m_meta = rx_meta_kv.match(line)
                if m_meta:
                    key = m_meta.group(1).strip().lower().replace(" ", "_")
                    val = m_meta.group(2).strip()
                    self.header_meta[key] = val

                m = rx_type.match(line)
                if m:
                    self.active_type = m.group(1).strip()
                    if (not started_geometry) and ("external perimeter" in self.active_type.lower()):
                        started_geometry = True
                    continue
                m = rx_width.match(line)
                if m:
                    self.active_width = float(m.group(1))
                    continue
                m = rx_height.match(line)
                if m:
                    self.active_height = float(m.group(1))
                    if self.layer_index >= 0:
                        layers[self.layer_index]["height"] = self.active_height
                    continue

                # slicer tag path: ;LAYER_CHANGE then ;Z:<val>
                if rx_layer_change.match(line):
                    pending_new_layer = True
                    continue
                mz = rx_z_tag.match(line)
                if mz:
                    zval = float(mz.group(1))
                    if pending_new_layer:
                        start_new_layer(zval, self.active_height or None)
                        pending_new_layer = False
                    continue

                if line.startswith("G20"):
                    self.units_mm = False
                    continue
                if line.startswith("G21"):
                    self.units_mm = True
                    continue
                if line.startswith("G90"):
                    self.abs_xyz = True
                    continue
                if line.startswith("G91"):
                    self.abs_xyz = False
                    continue
                if line.startswith("M82"):
                    self.abs_e = True
                    continue
                if line.startswith("M83"):
                    self.abs_e = False
                    continue
                if line.startswith("G92"):
                    parts = line.split()
                    for p in parts[1:]:
                        axis = p[0].upper()
                        try:
                            val = float(p[1:])
                        except Exception:
                            continue
                        if axis == "X":
                            self.x = val if self.abs_xyz else 0.0
                        elif axis == "Y":
                            self.y = val if self.abs_xyz else 0.0
                        elif axis == "Z":
                            self.z = val if self.abs_xyz else 0.0
                        elif axis == "E":
                            self.e = val if self.abs_e else 0.0
                            prev_e_for_abs = self.e
                            have_prev_e = True
                    continue

                # Both G0 and G1 move the nozzle, so both update the position and
                # can trigger a layer change; only G1 deposits material. Tracking
                # G0 matters for slicers that emit travels as G0 (PrusaSlicer uses
                # G1), where ignoring them would leave Z stale and turn the first
                # extrusion of a layer into a long spurious segment.
                if line.startswith("G0") or line.startswith("G1"):
                    is_extruding_move = line.startswith("G1")
                    parts = line.split()
                    got = {"X": None, "Y": None, "Z": None, "E": None, "F": None}
                    for p in parts[1:]:
                        axis = p[0].upper()
                        try:
                            val = float(p[1:])
                        except Exception:
                            continue
                        if axis in got:
                            got[axis] = val

                    if got["F"] is not None:
                        self.f = got["F"]

                    x0, y0 = self.x, self.y

                    def upd(old, val, is_abs):
                        return old if val is None else (val if is_abs else (old + val))

                    self.x = upd(self.x, got["X"], self.abs_xyz)
                    self.y = upd(self.y, got["Y"], self.abs_xyz)
                    self.z = upd(self.z, got["Z"], self.abs_xyz)
                    self.e = upd(self.e, got["E"], self.abs_e)

                    # If we have an established layer, detect Z increases and defer
                    # starting a new layer until the next extrusion.
                    if self.layer_index >= 0 and self.layer_z is not None and got["Z"] is not None:
                        if (self.z - float(self.layer_z)) > eps_z:
                            if self._pending_layer_z is None or self.z > self._pending_layer_z + eps_z:
                                self._pending_layer_z = float(self.z)

                    L = distance(x0, y0, self.x, self.y)
                    dE = 0.0
                    if got["E"] is not None:
                        if self.abs_e:
                            if have_prev_e:
                                dE = self.e - prev_e_for_abs
                            else:
                                dE = 0.0
                            prev_e_for_abs = self.e
                            have_prev_e = True
                        else:
                            dE = got["E"]

                    # A G0 that carries E is a purge or wipe, never geometry; the
                    # extruder position above stays in sync either way.
                    if not is_extruding_move:
                        dE = 0.0

                    if self.layer_index < 0 and dE > 0 and started_geometry:
                        start_new_layer(self.z, self.active_height or None)

                    # If a Z increase was detected and we now start extruding, we
                    # treat that as a new layer boundary.
                    if dE > 0 and self.layer_index >= 0 and self._pending_layer_z is not None and started_geometry:
                        if self.layer_z is None or (self._pending_layer_z - float(self.layer_z)) > eps_z:
                            start_new_layer(self._pending_layer_z, self.active_height or None)
                        self._pending_layer_z = None

                    if dE > 0 and self.layer_index >= 0 and started_geometry:
                        v_mm_s = (self.f / 60.0) if self.f > 0 else 0.0
                        width = self.active_width if self.active_width is not None else -1.0
                        height = self.active_height if self.active_height is not None else (self.layer_height or -1.0)
                        region_id = map_type_to_region(self.active_type)
                        seg = Segment(x0, y0, self.x, self.y, self.z, L, v_mm_s, dE, width, height, region_id)
                        layers[self.layer_index]["segs"].append(seg)
                    continue

                # else ignore

        last_h = None
        for idx in sorted(layers.keys()):
            if layers[idx]["height"] is None:
                layers[idx]["height"] = last_h
            else:
                last_h = layers[idx]["height"]

        return layers, self.header_meta


class Rasterizer:
    def __init__(self, px_mm: float, filament_diam: float, default_nozzle: float):
        self.px = px_mm
        self.Af = filament_area(filament_diam)
        self.default_nozzle = default_nozzle

    def _grid_from_segments(self, segs: List[Segment]) -> Tuple[int, int, Tuple[float, float]]:
        if not segs:
            return 1, 1, (0.0, 0.0)
        xs, ys = [], []
        pad = max(2 * self.default_nozzle, 1.0)
        for s in segs:
            xs += [s.x0, s.x1]
            ys += [s.y0, s.y1]
        minx = min(xs) - pad
        miny = min(ys) - pad
        maxx = max(xs) + pad
        maxy = max(ys) + pad
        W = max(1, int(math.ceil((maxx - minx) / self.px)))
        H = max(1, int(math.ceil((maxy - miny) / self.px)))
        return H, W, (minx, miny)

    def rasterize_layer(
        self,
        segs: List[Segment],
        z_mm: float,
        layer_h_mm: float,
        H: Optional[int] = None,
        W: Optional[int] = None,
        origin: Optional[Tuple[float, float]] = None,
    ):
        if H is None or W is None or origin is None:
            H, W, origin = self._grid_from_segments(segs)
        M = np.zeros((H, W), dtype=np.uint8)
        R = np.zeros((H, W), dtype=np.uint8)
        Q = np.zeros((H, W), dtype=np.float32)
        C = np.zeros((H, W), dtype=np.float32)

        for seg in segs:
            w = seg.width_mm if seg.width_mm > 0 else self.default_nozzle
            w = clamp(w, 0.2 * self.default_nozzle, 2.0 * self.default_nozzle)
            L = max(seg.L, 1e-6)
            v = max(seg.v_mm_s, 1e-6)
            dV = max(seg.dE_mm, 0.0) * self.Af
            Qseg = dV * v / L

            x0, y0 = origin
            cx0 = (seg.x0 - x0) / self.px
            cy0 = (seg.y0 - y0) / self.px
            cx1 = (seg.x1 - x0) / self.px
            cy1 = (seg.y1 - y0) / self.px
            r_px = max(1, int(math.ceil((0.5 * w) / self.px)))

            minx = clamp(int(math.floor(min(cx0, cx1))) - r_px - 1, 0, W - 1)
            maxx = clamp(int(math.ceil(max(cx0, cx1))) + r_px + 1, 0, W - 1)
            miny = clamp(int(math.floor(min(cy0, cy1))) - r_px - 1, 0, H - 1)
            maxy = clamp(int(math.ceil(max(cy0, cy1))) + r_px + 1, 0, H - 1)

            ux, uy = (cx1 - cx0), (cy1 - cy0)
            seg_len2 = max(ux * ux + uy * uy, 1e-8)

            for yy in range(miny, maxy + 1):
                py = yy + 0.5
                for xx in range(minx, maxx + 1):
                    px = xx + 0.5
                    t = ((px - cx0) * ux + (py - cy0) * uy) / seg_len2
                    if t < 0.0:
                        t = 0.0
                    elif t > 1.0:
                        t = 1.0
                    qx = cx0 + t * ux
                    qy = cy0 + t * uy
                    dx = px - qx
                    dy = py - qy
                    if (dx * dx + dy * dy) <= (r_px * r_px + 1e-6):
                        M[yy, xx] = 1
                        old = R[yy, xx]
                        if REGION_PRIORITY.get(seg.region_id, 0) >= REGION_PRIORITY.get(old, 0):
                            R[yy, xx] = seg.region_id
                        Q[yy, xx] += Qseg
                        C[yy, xx] += 1.0

        mask = C > 0
        Q[mask] /= C[mask]

        return LayerRaster(z_mm, layer_h_mm, H, W, self.px, origin, M, R, Q)


def gcode_to_gplan(
    gcode_path: str,
    out_dir: str,
    px_mm: float = None,
    filament_diam: float = None,
    default_nozzle: float = None,
    start_after_external_perimeter: bool = False,
):
    os.makedirs(out_dir, exist_ok=True)
    layers_dir = os.path.join(out_dir, "layers")
    os.makedirs(layers_dir, exist_ok=True)

    parser = GCodeParser(start_after_external_perimeter=start_after_external_perimeter)
    layers, header = parser.parse_file(gcode_path)

    # ---- Auto-detect filament diameter ----
    fd = None
    if "filament_diameter" in header:
        try:
            fd = float(str(header["filament_diameter"]).strip())
        except Exception:
            fd = None
    if fd is None:
        try:
            with open(gcode_path, "r", encoding="utf-8", errors="ignore") as fh:
                for ln in fh:
                    if ln.strip().startswith("M200"):
                        m = re.search(r"[dD]([0-9]*\.?[0-9]+)", ln)
                        if m:
                            fd = float(m.group(1))
                            break
        except Exception:
            fd = None
    if fd is None:
        fd = 1.75

    # ---- Auto-detect nozzle/line width ----
    nz = None
    if "nozzle_diameter" in header:
        try:
            nz = float(str(header["nozzle_diameter"]).strip())
        except Exception:
            nz = None

    widths = []
    for idx in layers:
        for seg in layers[idx]["segs"]:
            if seg.width_mm and seg.width_mm > 0:
                widths.append(seg.width_mm)
    if len(widths) >= 10:
        med_w = float(np.median(widths))
        if nz is None:
            nz = med_w
        else:
            nz = 0.5 * (nz + med_w)

    if nz is None:
        nz = 0.4

    # ---- Auto-select pixel size from line width ----
    if px_mm is None:
        px_mm = max(min(nz / 4.0, 0.25), 0.05)

    if filament_diam is None:
        filament_diam = fd
    if default_nozzle is None:
        default_nozzle = nz

    ras = Rasterizer(px_mm=px_mm, filament_diam=filament_diam, default_nozzle=default_nozzle)

    # ---- Compute global raster bounds so all layers share same H/W/origin ----
    xs_all, ys_all = [], []
    pad = max(2 * default_nozzle, 1.0)
    for idx in sorted(layers.keys()):
        for seg in layers[idx]["segs"]:
            xs_all += [seg.x0, seg.x1]
            ys_all += [seg.y0, seg.y1]
    if len(xs_all) > 0 and len(ys_all) > 0:
        minx = min(xs_all) - pad
        miny = min(ys_all) - pad
        maxx = max(xs_all) + pad
        maxy = max(ys_all) + pad
        W = max(1, int(math.ceil((maxx - minx) / px_mm)))
        H = max(1, int(math.ceil((maxy - miny) / px_mm)))
        origin_global = (minx, miny)
    else:
        H, W = 1, 1
        origin_global = (0.0, 0.0)

    gplan_layers = []
    for idx in sorted(layers.keys()):
        info = layers[idx]
        segs = info["segs"]
        z_mm = float(info["z"]) if info["z"] is not None else 0.0
        h_mm = float(info["height"]) if info["height"] is not None else 0.0

        lr = ras.rasterize_layer(segs, z_mm, h_mm, H=H, W=W, origin=origin_global)

        occ_path = os.path.join(layers_dir, f"L{idx:03d}_M.npy")
        reg_path = os.path.join(layers_dir, f"L{idx:03d}_R.npy")
        flow_path = os.path.join(layers_dir, f"L{idx:03d}_Q.npy")
        np.save(occ_path, lr.M)
        np.save(reg_path, lr.R)
        np.save(flow_path, lr.Q)

        gplan_layers.append(
            {
                "id": idx,
                "z_mm": lr.z_mm,
                "layer_height_mm": lr.height_mm,
                "raster": {
                    "H": int(lr.H),
                    "W": int(lr.W),
                    "px_mm": float(lr.px_mm),
                    "origin_xy_mm": [float(lr.origin_xy_mm[0]), float(lr.origin_xy_mm[1])],
                    "occupancy_uri": os.path.relpath(occ_path, out_dir),
                    "region_uri": os.path.relpath(reg_path, out_dir),
                    "flow_uri": os.path.relpath(flow_path, out_dir),
                },
            }
        )

    def parse_num(x):
        try:
            return float(x)
        except Exception:
            return None

    global_block = {
        "infill_density": None,
        "base_layer_height_mm": parse_num(header.get("layer_height", None)),
        "support_policy": "auto"
        if (header.get("support_material", "0").strip() not in ("0", "false", "False", "no", "None", ""))
        else "none",
    }
    if "fill_density" in header:
        m = re.match(r"(\d+)%", header["fill_density"].strip())
        if m:
            global_block["infill_density"] = int(m.group(1)) / 100.0

    manifest = {
        "schema_version": "0.3",
        "meta": {"units": "mm", "coords": "printer_xy_up_z", "source": "printanything.gplan.gcode_to_gplan"},
        "printer_profile": {
            "nozzle_diameter": parse_num(header.get("nozzle_diameter", default_nozzle)) or default_nozzle,
            "filament_diameter": filament_diam,
        },
        "material_profile": {
            "id": header.get("filament_type", "PLA"),
            "nozzle_temp_c": parse_num(header.get("temperature", None)),
            "bed_temp_c": parse_num(header.get("bed_temperature", None)),
        },
        "global": global_block,
        "layers": gplan_layers,
    }

    manifest_path = os.path.join(out_dir, MANIFEST_NAMES[0])
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest_path
