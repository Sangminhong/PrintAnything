# Data

## Slice-100K

Every experiment in the paper uses [Slice-100K](https://github.com/idealab-isu/slice-100k):
each sample is a CAD model (`.stl`) together with the G-code a standard slicing
pipeline produced for it. Objects are split 9:1 into train and test, and all
numbers are the mean over the held-out tenth.

Expected layout:

```
data/slice100k/
  stls/
    STLs_01/100072.stl
    STLs_02/100139.stl
    ...
  gcode/
    100072_objaverse_xl_config_4.gcode
    ...
```

A sample is identified by the G-code base name, e.g.
`100072_objaverse_xl_config_4`; the leading number before `_objaverse_xl_config_`
is the mesh id, which is looked up across the `STLs_*` subdirectories. An object
without both halves is skipped.

By default every indexed object is used. `--max_objects N` keeps only the first
N of them — the runs reported in the paper used `--max_objects 1000`, so pass it
whenever you compare against the tables (see [REPRODUCE.md](REPRODUCE.md)).

## Building the ground-truth G-plan maps

```bash
python tools/prepare_gplan.py \
    --gcode_root data/slice100k/gcode \
    --out_root   data/slice100k_gplan \
    --workers 8
```

Roughly 20 seconds per object on one core, so the parallel form is worth using.
Samples that already have a manifest are skipped, so the command is resumable.

The result is one folder per sample:

```
data/slice100k_gplan/100072_objaverse_xl_config_4/
  gplan_manifest.json
  layers/L000_M.npy     occupancy   (H, W) uint8
  layers/L000_R.npy     region      (H, W) uint8   0 gap · 1 wall · 2 infill · 3 support · 4 skirt
  layers/L000_Q.npy     flow        (H, W) float32
  ...
```

The manifest records, per slice, the physical height `z_mm`, the layer height,
and the raster frame (`H`, `W`, `px_mm`, `origin_xy_mm`) that maps pixels to
millimetres. All slices of one object share a frame, so the stack is a plain
`(Z, H, W)` array. `ir_manifest.json` is accepted as an alternative manifest
name, which is what the original research code wrote.

Pixel size defaults to a quarter of the detected extrusion width (typically
0.09–0.11 mm), so raw rasters are larger than the 256×256 the model works on.
The dataset loader resizes each slice onto that canvas by **downsample-to-fit and
centre-pad** — never by cropping, which would silently delete geometry — and
hands the exact raw→target mapping to the slice projection so the input is
aligned with the supervision pixel for pixel.

### Flow map normalisation

The raw flow map holds a physical volume rate per pixel. The loader normalises
it per slice (95th percentile, Gaussian smoothing, rescale to `[0, 1]`), which is
what the model regresses. That normalisation is not cheap; pass
`--q_cache_root <dir>` to cache it across epochs.

A consequence worth remembering: a flow map read straight from
`prepare_gplan.py` is *not* in `[0, 1]`, so compile it with `use_flow=False`
(`--no_flow`) unless it has been normalised.

## Your own point clouds

`tools/infer.py` takes a point cloud in millimetres and needs no ground truth:

```bash
python tools/infer.py --ckpt <ckpt> --input scan.ply --out scan.gcode --scale_to_mm 80
```

The object is centred and scaled to `[-1, 1]` internally, exactly as the dataset
does, and the number of slices follows its physical height divided by
`--layer_height_mm`.
