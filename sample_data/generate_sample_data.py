#!/usr/bin/env python3
"""
Generates a small pair of synthetic, colored point clouds for exercising
pointcloud_metrics.py without needing any real scan data.

reference.ply: points sampled on a unit sphere, colored by a simple
               gradient so the color-based metrics have something to chew on.
distorted.ply: the same sphere with added Gaussian noise on the geometry
               and a small perturbation on the colors, simulating a
               reconstruction that's close but not identical to the GT.
"""

import numpy as np
import open3d as o3d

RNG = np.random.default_rng(seed=0)
N_POINTS = 20000


def sample_sphere(n: int) -> np.ndarray:
    # Uniform sampling on the unit sphere via normalized Gaussian vectors.
    vecs = RNG.normal(size=(n, 3))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs


def colors_from_positions(points: np.ndarray) -> np.ndarray:
    # Map [-1, 1] coordinates to [0, 1] color channels.
    return np.clip((points + 1.0) / 2.0, 0.0, 1.0)


def make_point_cloud(points: np.ndarray, colors: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))
    return pcd


def main():
    ref_points = sample_sphere(N_POINTS)
    ref_colors = colors_from_positions(ref_points)
    reference = make_point_cloud(ref_points, ref_colors)

    dist_points = ref_points + RNG.normal(scale=0.01, size=ref_points.shape)
    dist_colors = np.clip(ref_colors + RNG.normal(scale=0.02, size=ref_colors.shape), 0.0, 1.0)
    distorted = make_point_cloud(dist_points, dist_colors)

    o3d.io.write_point_cloud("sample_data/reference.ply", reference)
    o3d.io.write_point_cloud("sample_data/distorted.ply", distorted)

    print(f"Wrote sample_data/reference.ply ({len(reference.points)} points)")
    print(f"Wrote sample_data/distorted.ply ({len(distorted.points)} points)")


if __name__ == "__main__":
    main()
