# Reproducing the paper

Every command below assumes the dataset has been prepared as described in
[DATA.md](DATA.md), and that the same dataset and split flags are passed to
every tool:

```bash
--stl_root data/slice100k/stls --gplan_root data/slice100k_gplan \
--val_split 0.1 --seed 42 --max_objects 1000
```

`--val_split`, `--seed` and `--max_objects` together define the held-out set.
Training, evaluation, G-code generation and the baselines all rebuild it with the
same seeded generator, so they score the same objects — change any of the three
and no number is comparable to any other.

**`--max_objects 1000` is what the reported numbers were produced with.** The
runs in the paper drew the 9:1 split from the first 1000 indexed objects, giving
900 training and 100 test objects; that is the `num_samples: 100` in the released
evaluation JSONs. The flag defaults to `0`, which uses every object that has both
a mesh and a G-plan map (4868 with the Slice-100K subset used here, i.e. a
4382/486 split) — a larger and perfectly reasonable setup, but its numbers are
not comparable with the tables below. Pass `--max_objects 1000` to reproduce
them.

## The metrics

| Symbol | What it measures | Where |
|---|---|---|
| CD | Chamfer distance between the predicted print surface and the GT mesh, Eq. (9) | `evaluation/geometry.py` |
| F1₃D | point F1 at τ = 1 mm, Eq. (10) | `evaluation/geometry.py` |
| F1₂D | the same point F1 computed per slice and averaged, Eq. (11) | `evaluation/geometry.py` |
| Δρ-smooth | mean \|ρ_k − ρ_{k−1}\| over extruding moves, Eq. (12) | `evaluation/flow.py` |
| ρ-CV | mean per-slice coefficient of variation of ρ, Eq. (13) | `evaluation/flow.py` |

Details that matter when comparing numbers:

* The predicted side is the **perimeter** of the occupancy map — the surface that
  actually gets printed — with skirt pixels removed; the GT side is 50k points
  sampled uniformly on the reference mesh.
* Only the **longest contour of each slice** is used, which is how the reported
  numbers were produced. On objects whose slices break into several islands this
  scores just the largest one while the ground truth still samples the whole
  mesh, so those objects look worse than they are; `--all_contours` scores every
  contour instead. Use one or the other consistently — the two are not
  comparable.
* Predictions live on the printer bed and the mesh lives in its own frame, so
  predictions are translated onto the mesh (XY raster centre → mesh bounding-box
  centre, bottom slice → lowest mesh point) before anything is measured. No
  rotation is applied.
* **CD is reported in normalised units**: both point sets are divided by the
  object's normalisation scale first, so CD does not grow with object size. F1₃D
  and F1₂D are computed in millimetres, where τ = 1 mm means what it says.
* Mesh-reconstruction baselines return a mesh in an arbitrary scale, so
  `tools/baselines/evaluate_gcode.py` additionally fits **one isotropic scale**
  before measuring, and reports the mean fitted scale. PrintAnything gets no such
  fit — it predicts the metric size directly.
* That script measures *rasterised toolpaths*, which are as wide as the extrusion
  width, so it inflates features thinner than a line width. It compares
  mesh-based pipelines with each other under one protocol; our own numbers come
  from `tools/evaluate.py`, which measures the G-plan map directly.
* The flow metrics depend on the compiler settings, so only compare runs
  compiled with the same `tools/generate_gcode.py` flags.

## Table 1 — comparison with mesh-based pipelines

Ours:

```bash
python tools/evaluate.py --ckpt runs/printanything/checkpoints/ep049.pt \
    --out_json outputs/metrics/ours.json
```

Baselines — reconstruct a mesh, slice it, score the G-code; see
[BASELINES.md](BASELINES.md):

```bash
python tools/baselines/export_point_clouds.py --out_dir outputs/baseline_inputs --format xyz
python tools/baselines/reconstruct_poisson.py --in_dir outputs/baseline_inputs \
    --out_dir outputs/poisson/meshes
python tools/baselines/slice_meshes.py --mesh_dir outputs/poisson/meshes \
    --out_dir outputs/poisson/gcode --config <prusaslicer profile>.ini
python tools/baselines/evaluate_gcode.py --gcode_root outputs/poisson/gcode \
    --out_json outputs/metrics/poisson.json
```

## Table 2 — multi-slice conditioning

`--z_context` is the number of neighbouring slabs used per side; `0` disables
MSC, `1` is the paper's setting.

```bash
python tools/train.py --out_root runs/msc_off --z_context 0
python tools/train.py --out_root runs/msc_on  --z_context 1
python tools/evaluate.py --ckpt runs/msc_off/checkpoints/ep049.pt
python tools/evaluate.py --ckpt runs/msc_on/checkpoints/ep049.pt
```

## Table 3 — infill policy recommendation

```bash
python tools/build_infill_dataset.py --ckpt <ckpt> \
    --out_csv outputs/infill/trials.csv --num_objects 200 --trials_per_object 12
python tools/train_infill_recommender.py --csv outputs/infill/trials.csv \
    --out_dir outputs/infill/recommender
```

The second command prints strength, cost and the combined score
`1e5 × S / C` for the recommender and for each fixed-pattern baseline, on
objects the surrogate never saw (the split is by object id, not by row).

## Table 4 — G-plan map design

| Row | Train flags |
|---|---|
| M | `--no_region --no_flow` |
| M + R | `--no_flow` |
| M + R + Q | *(defaults)* |

```bash
python tools/train.py --out_root runs/abl_M   --no_region --no_flow
python tools/train.py --out_root runs/abl_MR  --no_flow
python tools/train.py --out_root runs/abl_MRQ
```

Geometry columns come from `tools/evaluate.py`. The flow columns are measured on
compiled toolpaths, so generate the G-code first:

```bash
python tools/generate_gcode.py --ckpt runs/abl_MRQ/checkpoints/ep049.pt \
    --out_dir outputs/abl_MRQ/gcode
python tools/flow_metrics.py --gcode_root outputs/abl_MRQ/gcode
```

## Sec. 5.6 — robustness to scan artifacts

Two augmentations are applied during training:

```bash
python tools/train.py --out_root runs/robust \
    --noise_every 1 --noise_sigma 0.01 \
    --hole_every 1  --hole_radius 0.12 --hole_frac 0.06
```

`--noise_sigmas` / `--hole_fracs` take a comma-separated list, drawn from per
firing, to cover several corruption levels with one model. `--simulated_scans`
replaces the uniform mesh samples with partial multi-view depth scans
altogether.

To evaluate under corruption without retraining, export corrupted point clouds
with `tools/baselines/export_point_clouds.py --noise_sigma <s>` and run them
through `tools/infer.py`.

## Differences from the original research code

The released code was reorganised around the structure of the paper. Two changes
alter behaviour and are worth stating plainly:

1. **Serpentine infill on alternating slices.** In the original compiler the
   direction flip on odd slices left the scan producing no per-island spans, so
   odd slices came out with perimeters only. The released compiler emits infill
   on every slice, reversed on odd ones. Toolpath-dependent numbers (the flow
   metrics, total filament) therefore differ from the original runs; the
   geometry metrics, which are computed on the G-plan maps, do not.
2. **First slice height.** A predicted G-plan folder now starts at one layer
   height above the bed instead of `Z0`.

The flow loss is selectable: `--flow_loss huber` (default) is the masked Huber of
Eq. (7); `--flow_loss l1_sum --lambda_Q 3e-5` reproduces the unnormalised L1 the
original runs used.
