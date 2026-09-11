# PrintAnything

**Learning Geometric Plan Map for 3D Printing G-code Generation from Unoriented Point Clouds**

Sangmin Hong, Daniel Sungho Jung, Heewon Kim, Kyoung Mu Lee — ECCV 2026

[paper](https://arxiv.org/abs/2607.27729) · [arXiv:2607.27729](https://arxiv.org/abs/2607.27729)

PrintAnything turns a raw, unoriented point cloud directly into executable 3D
printing G-code — no mesh reconstruction anywhere in the pipeline. Instead of
repairing a reconstructed surface, it predicts a **Geometric plan (G-plan) map**:
a compact per-slice representation of what to print, made of an occupancy map
`M`, a region map `R` and a flow map `Q`.

```
point cloud ──► slice-wise projection ──► GPNet ──► G-plan map ──► infill ──► G-code
                    (Sec. 3.2)          (Sec. 3.2)  M, R, Q     (Sec. 3.3)  (Sec. 3.4)
```

---

## Install

```bash
conda create -n printanything python=3.10 && conda activate printanything
pip install -r requirements.txt
```

PyTorch should be installed for your own CUDA version first
(see [pytorch.org](https://pytorch.org)). Everything runs from a plain checkout —
no `pip install -e .` needed; the tools add the repository root to `sys.path`
themselves.

The default point encoder (`--point_encoder ptv3`) additionally needs the Point
Transformer V3 dependencies — `addict`, `timm`, `torch-scatter` and a `spconv`
build matching your CUDA version. `--point_encoder pointnet` needs nothing
beyond PyTorch.

Optional extras: `open3d` (Poisson baseline), `scikit-learn` (infill
recommender), `rtree` (infill inside-mesh validity check).

A quick check that the installation works, no dataset required:

```bash
python tests/test_pipeline.py
```

## Quick start — print your own point cloud

```bash
python tools/infer.py \
    --ckpt      checkpoints/printanything.pt \
    --input     my_scan.ply \
    --out       my_scan.gcode \
    --scale_to_mm 80 \
    --bed_center_xy_mm 110,110
```

Accepts `.ply`, `.xyz`, `.npy` and any mesh trimesh can read (it is sampled).
Normals are never used. **Always preview the result in a slicer before
printing**, and check that the printing parameters on the command line match
your machine.

## Repository layout

```
printanything/
  gplan/            the G-plan map
    constants.py        M / R / Q, the region classes, the raster size
    gcode_to_gplan.py   G-code  -> ground-truth G-plan map        (Sec. 3.1)
    slice_projection.py point cloud -> slice-aligned 2D input     (Sec. 3.2)
    gplan_to_gcode.py   G-plan map -> executable G-code           (Sec. 3.4)
    io.py               reading / writing G-plan folders
  models/
    point_encoders.py   E_global: Point Transformer V3 or PointNet-style MLP
    unet.py             U-Net blocks and FiLM, Eq. (1)
    gpnet.py            GPNet: the G-plan map predictor           (Sec. 3.2)
  infill/
    patterns.py         template dictionary + periodic warp/insert
    proxies.py          strength and cost proxies
    recommender.py      the (pattern, scale) surrogate + Pareto pick (Sec. 3.3)
  engine/
    losses.py           Eqs. (5)-(8)
    trainer.py          train / validate loops
  evaluation/
    geometry.py         CD, F1_3D, F1_2D, Eqs. (9)-(11)
    flow.py             d(rho)-smooth, rho-CV, Eqs. (12)-(13)
  datasets/             Slice-100K, the 9:1 split, scan augmentations
  inference.py          point cloud -> G-plan map -> G-code

tools/                  command line entry points (see below)
scripts/                thin shell wrappers around the usual runs
tests/                  fast end-to-end checks (no dataset, no GPU)
assets/infill_patterns/ the infill template dictionary
third_party/            Point Transformer V3 (original licence kept)
docs/                   DATA.md, REPRODUCE.md, BASELINES.md
```

## Data

PrintAnything trains on [Slice-100K](https://github.com/idealab-isu/slice-100k),
which pairs a CAD model with the G-code a standard slicer produced for it. The
ground-truth G-plan maps are rasterised from that G-code once:

```bash
python tools/prepare_gplan.py \
    --gcode_root data/slice100k/gcode \
    --out_root   data/slice100k_gplan \
    --workers 8
```

See [docs/DATA.md](docs/DATA.md) for the expected directory layout and the
on-disk format.

## Train

```bash
python tools/train.py \
    --stl_root   data/slice100k/stls \
    --gplan_root data/slice100k_gplan \
    --out_root   runs/printanything
```

Defaults follow the paper: 30k input points, 256×256 maps, AdamW at 2e-4 with
weight decay 1e-4, mixed precision, 50 epochs, one object per step. The split is
9:1 with `--seed 42`; every other tool rebuilds the identical split, so the
held-out objects are always the same.

The runs reported in the paper drew that split from the first 1000 indexed
objects — add `--max_objects 1000` here and to every evaluation command to
reproduce the tables, and see [docs/REPRODUCE.md](docs/REPRODUCE.md).

Ablations: `--z_context 0` (no multi-slice conditioning, Table 2), `--no_region`
(M only) and `--no_flow` (M + R) for Table 4.

## Evaluate

```bash
python tools/evaluate.py --ckpt runs/printanything/checkpoints/ep049.pt
```

Reports CD, F1₃D, F1₂D and slice IoU over the held-out objects. To score the
generated toolpaths as well:

```bash
python tools/generate_gcode.py --ckpt <ckpt> --out_dir outputs/val_gcode
python tools/flow_metrics.py   --gcode_root outputs/val_gcode
```

[docs/REPRODUCE.md](docs/REPRODUCE.md) maps each table of the paper to the exact
command, and [docs/BASELINES.md](docs/BASELINES.md) covers the mesh-based
baselines (Poisson, DWG, MeshAnything).

## Infill recommender

```bash
python tools/build_infill_dataset.py --ckpt <ckpt> --out_csv outputs/infill/trials.csv
python tools/train_infill_recommender.py --csv outputs/infill/trials.csv \
    --out_dir outputs/infill/recommender
```

The first command synthesises random `(pattern, scale)` policies on predicted
G-plan maps and scores them with the strength/cost proxies; the second fits the
surrogate and compares it against the fixed-policy baselines of Table 3.

## Notes on the released code

* **Point encoder.** `--point_encoder ptv3` (default) is the Point Transformer V3
  encoder described in the paper. `--point_encoder pointnet` is a light
  PointNet-style encoder that needs no custom kernels. A checkpoint remembers
  which one it used, so `load_gpnet_checkpoint` picks it up automatically.
* **Flow map range.** The flow head is an unbounded regressor, and its prediction
  is clamped to `[0, 1]` — the range of the ground-truth flow map — before
  compilation. Where the model predicts near-zero flow the compiler emits a
  toolpath that deposits little or no material; pass `--no_flow` to fall back to
  the purely geometric extrusion `line_width × layer_height × length`.
* **Printing parameters** (`--line_width`, `--walls`, `--infill_density`,
  speeds, `--bed_center_xy_mm`) are compiler settings, not model outputs. Set
  them for your own printer.

## Citation

```bibtex
@inproceedings{hong2026printanything,
  title     = {PrintAnything: Learning Geometric Plan Map for 3D Printing G-code
               Generation from Unoriented Point Clouds},
  author    = {Hong, Sangmin and Jung, Daniel Sungho and Kim, Heewon and Lee, Kyoung Mu},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

## Licence

Released under [CC BY-NC 4.0](LICENSE) (non-commercial).
`third_party/PointTransformerV3` keeps its original licence.
