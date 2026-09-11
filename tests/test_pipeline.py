#!/usr/bin/env python3
"""Fast end-to-end checks that need no dataset and no GPU.

    python tests/test_pipeline.py        # or: pytest tests/test_pipeline.py

Covers the seams between the stages of Figure 2: the slice projection, a GPNet
forward pass, the losses, the G-plan folder round trip, the G-code compiler, the
flow metrics, and infill synthesis with its proxies.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from printanything.engine import LossConfig, occupancy_loss, region_loss  # noqa: E402
from printanything.evaluation import (  # noqa: E402
    RasterFrame,
    chamfer_distance,
    gcode_flow_metrics,
    occupancy_points,
    perimeter_points,
    point_f1,
    slice_iou_f1,
)
from printanything.gplan import (  # noqa: E402
    GCodeConfig,
    INFILL,
    WALL,
    gcode_to_gplan,
    gplan_to_gcode,
    load_gplan_stack,
    num_projection_channels,
    project_slice,
    save_gplan_folder,
)
from printanything.infill import (  # noqa: E402
    PATTERNS,
    InfillPolicy,
    InfillRecommender,
    apply_infill,
    combined_score,
    compute_proxies,
)
from printanything.models import GPNet  # noqa: E402


def cylinder_gplan(Z=6, H=48, W=48, radius=16):
    """A hollow cylinder: walls on the boundary ring, infill inside."""
    yy, xx = np.mgrid[0:H, 0:W]
    dist = np.hypot(yy - H / 2, xx - W / 2)
    R = np.zeros((Z, H, W), np.uint8)
    R[:, dist < radius] = INFILL
    R[:, (dist < radius) & (dist >= radius - 2)] = WALL
    M = (R > 0).astype(np.uint8)
    Q = np.where(M > 0, 0.8, 0.0).astype(np.float32)
    return M, R, Q


def test_slice_projection():
    pc = torch.rand(2000, 3) * 2 - 1
    for z_context in (0, 1, 2):
        out = project_slice(pc, 0.5, 0.1, 32, 32, device=torch.device("cpu"), z_context=z_context)
        assert out.shape == (1, num_projection_channels(z_context), 32, 32)
        assert torch.isfinite(out).all()
    # An empty cloud must not crash the decoder input.
    assert project_slice(torch.zeros(0, 3), 0.5, 0.1, 8, 8, device=torch.device("cpu")).sum() == 0
    print("ok  slice projection")


def test_gpnet_forward():
    model = GPNet(point_encoder="pointnet", global_dim=64, z_context=1)
    pc = torch.rand(1, 500, 3) * 2 - 1
    M, R, Q = model(pc, torch.linspace(0, 1, 4), [(32, 32)] * 4, slice_batch_size=2)
    assert len(M) == len(R) == len(Q) == 4
    assert M[0].shape == (1, 1, 32, 32) and R[0].shape == (1, 5, 32, 32)

    # M is derived from R, so a gap pixel can never be occupied.
    prob_gap = torch.softmax(R[0], dim=1)[:, 0:1]
    assert torch.allclose(torch.sigmoid(M[0]), 1.0 - prob_gap, atol=1e-4)

    # Turning off the heads must change nothing else.
    lean = GPNet(point_encoder="pointnet", global_dim=64, num_classes_R=0, predict_Q=False)
    M2, R2, Q2 = lean(pc, torch.linspace(0, 1, 2), [(32, 32)] * 2)
    assert R2 == [] or R2 is None
    assert not Q2 and M2[0].shape == (1, 1, 32, 32)
    print("ok  GPNet forward")


def test_losses():
    logit = torch.randn(2, 1, 16, 16, requires_grad=True)
    target = (torch.rand(2, 1, 16, 16) > 0.5).float()
    loss = occupancy_loss(logit, target)
    loss.backward()
    assert loss.item() > 0 and logit.grad is not None

    region = torch.randn(2, 5, 16, 16)
    labels = torch.randint(0, 5, (2, 16, 16))
    assert region_loss(region, labels, [0.1, 3.0, 1.0, 2.0, 0.5]).item() > 0

    cfg = LossConfig()
    mask = torch.ones(2, 1, 16, 16)
    assert cfg.flow(torch.zeros(2, 1, 16, 16), torch.ones(2, 1, 16, 16), mask).item() > 0
    print("ok  losses")


def test_gplan_roundtrip_and_gcode():
    M, R, Q = cylinder_gplan()
    with tempfile.TemporaryDirectory() as tmp:
        manifest = save_gplan_folder(tmp, M, R, Q, px_mm=0.5, dz_mm=0.2, sample_id="cylinder")
        M2, R2, Q2, meta = load_gplan_stack(manifest)
        assert np.array_equal(M, M2) and np.array_equal(R, R2)
        assert np.allclose(Q, Q2) and abs(meta["dz_mm"] - 0.2) < 1e-6
        # The first slice must clear the bed.
        assert meta["z_start_mm"] > 0

        gcode = gplan_to_gcode(tmp, str(Path(tmp) / "out.gcode"),
                               GCodeConfig(walls=2, infill_density=1.0), progress=False)
        text = Path(gcode).read_text()
        assert ";TYPE:PERIMETER" in text and ";TYPE:INFILL" in text
        # Every slice carries toolpaths, in both serpentine directions.
        assert text.count(";LAYER:") == M.shape[0]
        for z in range(M.shape[0]):
            body = text.split(f";LAYER:{z}\n")[1].split(";LAYER:")[0]
            assert "G1 " in body.split(";TYPE:INFILL")[1], f"slice {z} has no infill"

        flow = gcode_flow_metrics(gcode)
        assert flow["num_moves"] > 0 and np.isfinite(flow["rho_smooth"])
    print("ok  G-plan round trip and G-code compilation")


def test_gcode_roundtrip_preserves_geometry():
    """Compile a G-plan map and rasterise the G-code back: same slices, same size.

    Both directions have to agree on millimetres, or a predicted object would be
    printed at the wrong scale. The border-touching case is included on purpose:
    such a slice has no contour, so the compiler has to announce its height even
    when it emits no perimeter.
    """
    px, dz, Z = 0.2, 0.3, 5
    H, W = 60, 100
    origin = (100.0, 50.0)

    def roundtrip(M, R):
        with tempfile.TemporaryDirectory() as tmp:
            save_gplan_folder(tmp + "/src", M, R, px_mm=px, dz_mm=dz, origin_xy_mm=origin)
            gcode = gplan_to_gcode(tmp + "/src", tmp + "/out.gcode",
                                   GCodeConfig(walls=2, infill_density=1.0, use_flow=False),
                                   progress=False)
            M2, R2, _, meta = load_gplan_stack(gcode_to_gplan(gcode, tmp + "/back"))

        src = occupancy_points(M, RasterFrame(origin_xy_mm=origin, px_mm=px, dz_mm=dz, z_start_mm=dz))
        back = occupancy_points(
            M2,
            RasterFrame(origin_xy_mm=meta["origin_xy_mm"], px_mm=meta["px_mm_raw"],
                        dz_mm=meta["dz_mm"], z_start_mm=meta["z_start_mm"]),
        )
        assert M2.shape[0] == M.shape[0], f"{M.shape[0]} slices in, {M2.shape[0]} out"
        size_in, size_out = src.max(0) - src.min(0), back.max(0) - back.min(0)
        assert np.allclose(size_in, size_out, atol=4 * px), f"{size_in} vs {size_out}"

    # Fills the whole raster: no contour, so no perimeter is emitted.
    R = np.full((Z, H, W), INFILL, np.uint8)
    R[:, :2, :] = WALL
    roundtrip(np.ones((Z, H, W), np.uint8), R)

    # The ordinary case, with a margin around the object.
    M = np.zeros((Z, H, W), np.uint8)
    M[:, 10:-10, 10:-10] = 1
    R = np.zeros((Z, H, W), np.uint8)
    R[:, 10:-10, 10:-10] = INFILL
    R[:, 10:12, 10:-10] = WALL
    roundtrip(M, R)
    print("ok  G-code round trip preserves slices and size")


def test_metrics():
    M, R, _ = cylinder_gplan()
    frame = RasterFrame(px_mm=0.5, dz_mm=0.2)
    pts = perimeter_points(M, frame, R=R)
    assert pts.shape[1] == 3 and pts.shape[0] > 0

    assert chamfer_distance(pts, pts) < 1e-9
    assert point_f1(pts, pts, tau_mm=1.0)[2] == 1.0

    iou, f1 = slice_iou_f1(M, M)
    assert abs(iou - 1.0) < 1e-9 and abs(f1 - 1.0) < 1e-9

    shifted = M.copy()
    shifted[:, :-4] = M[:, 4:]
    assert slice_iou_f1(shifted, M)[0] < 1.0
    print("ok  metrics")


def test_infill():
    M, R, _ = cylinder_gplan()
    for pattern in PATTERNS:
        M_inf, R_inf = apply_infill(M, R, pattern=pattern, scale=0.4)
        # Walls survive the infill synthesis untouched.
        assert np.array_equal(R_inf == WALL, R == WALL)
        # Infill never leaks outside the region the model designated.
        assert not np.any((R_inf == INFILL) & (R == 0))
        proxies = compute_proxies(M_inf, R_inf)
        assert 0.0 <= proxies["strength_proxy"] <= 1.0 and proxies["cost_time"] > 0
        assert combined_score(proxies["strength_proxy"], proxies["cost_time"]) > 0

    rows = []
    for pattern in PATTERNS:
        for scale in (0.3, 0.7, 1.2):
            p = compute_proxies(*apply_infill(M, R, pattern=pattern, scale=scale))
            rows.append({"pattern": pattern, "scale": scale, "layer_idx": 0, **p})

    try:
        recommender = InfillRecommender().fit(rows)
    except ImportError:
        print("skip infill recommender (scikit-learn not installed)")
        return

    candidates = [InfillPolicy(p, s) for p in PATTERNS for s in (0.3, 0.7, 1.2)]
    best, info = recommender.recommend(candidates, layer_idx=0)
    assert best in candidates and info["pareto_size"] >= 1
    print("ok  infill synthesis, proxies and recommender")


def main() -> int:
    torch.manual_seed(0)
    np.random.seed(0)
    for test in (
        test_slice_projection,
        test_gpnet_forward,
        test_losses,
        test_gplan_roundtrip_and_gcode,
        test_gcode_roundtrip_preserves_geometry,
        test_metrics,
        test_infill,
    ):
        test()
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
