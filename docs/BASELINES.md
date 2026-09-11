# Mesh-based baselines

The comparison of Table 1 is against the workflow PrintAnything replaces:

```
point cloud ──► mesh reconstruction ──► PrusaSlicer ──► G-code
```

Three reconstruction methods are compared — classical Poisson, and the learned
DWG and MeshAnything. All three must see the same input as PrintAnything, which
`tools/baselines/export_point_clouds.py` guarantees: the same held-out objects,
30k surface samples, and normals estimated from the points by k-NN PCA (never
taken from the mesh — the paper's setting is *unoriented* point clouds).

## 1. Export the inputs

```bash
python tools/baselines/export_point_clouds.py \
    --out_dir outputs/baseline_inputs --format xyz
```

`--format xyz` writes `x y z nx ny nz` per line (DWG, and the Poisson tool here),
`--format pwn` is the same content under the extension the Poisson binary
expects, `--format ply` suits methods that read point clouds directly.
`--noise_sigma` adds Gaussian noise relative to the object's bounding-box
diagonal, for the robustness study.

## 2. Reconstruct

**Poisson** runs here, headless, through Open3D:

```bash
python tools/baselines/reconstruct_poisson.py \
    --in_dir outputs/baseline_inputs --out_dir outputs/poisson/meshes --depth 9
```

Vertices below the 2nd percentile of the density estimate are trimmed; without
that, Poisson closes sparsely observed regions with a large bubble that a slicer
would cheerfully fill with material.

**DWG** ([Diffusing Winding Gradients](https://github.com/jsnln/DWG)) and
**MeshAnything** ([MeshAnything](https://github.com/buaacyw/MeshAnything)) are run
from their own repositories, on the exported point clouds, following their
instructions. Point the next step at whatever directory of meshes they produce.

## 3. Slice

```bash
python tools/baselines/slice_meshes.py \
    --mesh_dir outputs/poisson/meshes \
    --out_dir  outputs/poisson/gcode \
    --prusa_slicer /usr/bin/prusa-slicer \
    --config configs/prusa_slice100k.ini
```

Use the profile the reference G-code of Slice-100K was produced with, otherwise
the comparison mixes geometry errors with profile differences. Newer PrusaSlicer
builds emit binary `.bgcode`; pass `--bgcode_bin` (libbgcode's `bgcode` tool) and
the files are converted to plain `.gcode`.

Meshes the slicer rejects are reported and skipped. That count is itself a
result: a reconstruction with inverted faces or a non-manifold boundary can fail
to slice at all, which is the failure mode Sec. 1 describes.

## 4. Score

```bash
python tools/baselines/evaluate_gcode.py \
    --gcode_root outputs/poisson/gcode \
    --out_json   outputs/metrics/poisson.json
```

The G-code is rasterised back into a G-plan map with the same
`gcode_to_gplan` used for the ground truth, so exactly the same metrics apply as
in `tools/evaluate.py`. Reconstruction methods return a mesh in an arbitrary
scale, so one isotropic scale is fitted to the ground truth before measuring
(`--no_isotropic_fit` turns that off) and the mean fitted scale is reported —
a value far from 1.0 says the baseline got the object's size wrong.

Report how many objects each baseline actually produced G-code for
(`num_missing_gcode`, `num_failed` in the JSON) alongside its metrics: averages
over different subsets are not directly comparable.
