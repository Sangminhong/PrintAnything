#!/usr/bin/env python3
"""Poisson surface reconstruction baseline (Table 1).

Screened Poisson reconstruction over the exported point clouds, via Open3D, so
the whole baseline runs headless:

    python tools/baselines/export_point_clouds.py --out_dir outputs/pc --format xyz
    python tools/baselines/reconstruct_poisson.py --in_dir outputs/pc --out_dir outputs/poisson/meshes
    python tools/baselines/slice_meshes.py --mesh_dir outputs/poisson/meshes --out_dir outputs/poisson/gcode
    python tools/baselines/evaluate_gcode.py --gcode_root outputs/poisson/gcode

Low-density vertices are trimmed away, otherwise Poisson closes the mesh with a
large bubble around sparsely observed areas, which a slicer would happily fill.

DWG and MeshAnything are run from their own repositories on the very same
exported point clouds; see ``docs/BASELINES.md``.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm


def load_points_with_normals(path: Path):
    """Read an exported ``.xyz`` / ``.pwn`` file (x y z nx ny nz)."""
    data = np.loadtxt(path, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 6:
        raise ValueError(f"{path} has {data.shape[1]} columns; 6 (xyz + normals) are required")
    return data[:, :3], data[:, 3:6]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in_dir", required=True, help="Directory of exported point clouds.")
    ap.add_argument("--out_dir", required=True, help="Where to write the reconstructed meshes.")
    ap.add_argument("--glob", default="*.xyz")
    ap.add_argument("--depth", type=int, default=9, help="Octree depth of the Poisson solver.")
    ap.add_argument("--density_quantile", type=float, default=0.02,
                    help="Trim vertices below this quantile of the density estimate (0 = keep all).")
    ap.add_argument("--ext", default="ply", choices=("ply", "obj", "stl"))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    try:
        import open3d as o3d
    except ImportError:
        sys.exit("Open3D is required: pip install open3d")
    import trimesh

    files = sorted(Path(args.in_dir).glob(args.glob))
    if not files:
        sys.exit(f"No files matching '{args.glob}' under {args.in_dir}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    done = failed = 0

    for path in tqdm(files, desc="poisson", ncols=100):
        out_path = out_dir / f"{path.stem}.{args.ext}"
        if out_path.exists() and not args.overwrite:
            continue
        try:
            points, normals = load_points_with_normals(path)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.normals = o3d.utility.Vector3dVector(normals)

            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=args.depth
            )
            if args.density_quantile > 0:
                densities = np.asarray(densities)
                mesh.remove_vertices_by_mask(densities < np.quantile(densities, args.density_quantile))

            trimesh.Trimesh(
                vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.triangles)
            ).export(out_path)
            done += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  fail {path.name}: {exc}", file=sys.stderr)

    print(f"[done] {done} meshes in {out_dir} ({failed} failures)")


if __name__ == "__main__":
    main()
