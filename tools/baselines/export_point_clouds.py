#!/usr/bin/env python3
"""Export the held-out point clouds for the mesh-reconstruction baselines.

The baselines of Table 1 (Poisson, DWG, MeshAnything) take a point cloud and
return a mesh. They must see exactly the same input as PrintAnything: the same
objects, the same 30k surface samples, and -- because the paper studies
*unoriented* point clouds -- normals estimated from the points themselves by
k-NN PCA, never taken from the mesh.

    python tools/baselines/export_point_clouds.py --out_dir outputs/baseline_inputs --format xyz

Formats: ``xyz`` (``x y z nx ny nz``, for DWG), ``pwn`` (same columns, for the
Poisson binary), ``ply`` (points only, for learned methods). ``--noise_sigma``
adds Gaussian noise relative to the object size for the robustness study of
Sec. 5.6.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from _common import add_data_args, dataset_and_split  # noqa: E402
from printanything.utils import set_seed  # noqa: E402


def estimate_normals(points: np.ndarray, k: int = 30) -> np.ndarray:
    """Unoriented normals by local PCA, flipped to point away from the centroid.

    The smallest-eigenvalue eigenvector of the local covariance is the surface
    normal; its sign is arbitrary, so it is oriented outwards, which is what the
    reconstruction methods expect.
    """
    from scipy.spatial import cKDTree

    points = np.asarray(points, np.float64).reshape(-1, 3)
    n = points.shape[0]
    if n < 4:
        return np.tile(np.array([0.0, 0.0, 1.0]), (n, 1))

    k_use = min(k, n - 1)
    _, idx = cKDTree(points).query(points, k=k_use + 1)
    neighbours = points[np.atleast_2d(idx)]
    centered = neighbours - neighbours.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", centered, centered)

    try:
        normals = np.linalg.eigh(cov)[1][:, :, 0].copy()
    except np.linalg.LinAlgError:
        normals = np.zeros((n, 3))
        normals[:, 2] = 1.0
        return normals

    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.where(norm < 1e-9, 1.0, norm)
    outward = np.sum(normals * (points - points.mean(axis=0)), axis=1) < 0
    normals[outward] *= -1
    return normals


def sample_surface(stl_path: str, num_points: int) -> np.ndarray:
    import trimesh

    mesh = trimesh.load(stl_path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        if isinstance(mesh, trimesh.Scene) and len(mesh.geometry) > 0:
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        else:
            raise ValueError(f"Could not load a mesh from {stl_path}")
    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    return np.asarray(points, np.float64).reshape(-1, 3)


def write_points(path: Path, points: np.ndarray, normals: np.ndarray, fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt in ("xyz", "pwn"):
        np.savetxt(path, np.hstack([points, normals]), fmt="%.6f")
    elif fmt == "ply":
        with open(path, "w") as fh:
            fh.write("ply\nformat ascii 1.0\n")
            fh.write(f"element vertex {points.shape[0]}\n")
            fh.write("property float x\nproperty float y\nproperty float z\n")
            fh.write("property float nx\nproperty float ny\nproperty float nz\n")
            fh.write("end_header\n")
            for p, n in zip(points, normals):
                fh.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
    else:
        raise ValueError(f"Unknown format '{fmt}'")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_args(ap)
    ap.add_argument("--out_dir", required=True, help="Destination directory.")
    ap.add_argument("--format", default="xyz", choices=("xyz", "pwn", "ply"))
    ap.add_argument("--knn", type=int, default=30, help="Neighbours used for normal estimation.")
    ap.add_argument("--noise_sigma", type=float, default=0.0,
                    help="Gaussian noise, as a fraction of the object's bounding-box diagonal.")
    ap.add_argument("--max_samples", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    dataset, _, val_set = dataset_and_split(args, use_R=False, use_Q=False)
    indices = list(val_set.indices)
    if args.max_samples:
        indices = indices[: args.max_samples]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    written = 0

    for idx in tqdm(indices, desc="export", ncols=100):
        entry = dataset.samples[idx]
        sample_id = str(entry["id"]).replace("/", "_")
        try:
            points = sample_surface(entry["stl_path"], args.num_points)
            if args.noise_sigma > 0:
                diagonal = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
                points = points + rng.normal(0.0, args.noise_sigma * diagonal, points.shape)
            normals = estimate_normals(points, k=args.knn)
            write_points(out_dir / f"{sample_id}.{args.format}", points, normals, args.format)
            written += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {sample_id}: {exc}", file=sys.stderr)

    print(f"[done] {written}/{len(indices)} point clouds written to {out_dir}")


if __name__ == "__main__":
    main()
