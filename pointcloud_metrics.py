#!/usr/bin/env python3
"""
Point Cloud Comparison and Metrics Evaluation

This script compares two point clouds and computes various metrics:
- Chamfer Distance (CD)
- Hausdorff Distance
- Precision, Recall, F-score
- Point-to-point RMSE
- Point-to-Plane PSNR MSE D2
- Point-to-Attribute PSNR MSE YUV
- Point Cloud Structural Similarity (PointSSIM)
- PCQM (Point Cloud Quality Metric)
- MS-GraphSIM (Multi-Scale Graph Similarity)
"""

import numpy as np
import argparse
from pathlib import Path
from typing import Tuple, Dict, Optional
import sys
from scipy.spatial import cKDTree
import warnings
import open3d as o3d


class PointCloud:
    """Container for point cloud data with geometry, colors, and normals."""
    def __init__(self, points: np.ndarray, colors: Optional[np.ndarray] = None, 
                 normals: Optional[np.ndarray] = None):
        self.points = points  # (N, 3)
        self.colors = colors  # (N, 3) RGB in [0, 255]
        self.normals = normals  # (N, 3)
    
    def __len__(self):
        return len(self.points)


def load_point_cloud(filepath: str, load_colors: bool = True, 
                    load_normals: bool = False,
                    voxel_size: Optional[float] = None,
                    max_points: Optional[int] = None) -> PointCloud:
    """
    Load a point cloud from various file formats (.ply, .pcd, .xyz, .npy)
    
    Args:
        filepath: Path to the point cloud file
        load_colors: Whether to load color information
        load_normals: Whether to load normal information
        voxel_size: If set, downsample using voxel grid with this leaf size
                    (only applied to .ply/.pcd via Open3D)
        max_points: If set, randomly downsample to at most this many points
    
    Returns:
        PointCloud object with geometry and optional colors/normals
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Point cloud file not found: {filepath}")
    
    colors = None
    normals = None
    
    if filepath.suffix == '.npy':
        data = np.load(filepath)
        points = data[:, :3]
        if load_colors and data.shape[1] >= 6:
            colors = data[:, 3:6]
        if load_normals and data.shape[1] >= 9:
            normals = data[:, 6:9]
    elif filepath.suffix in ['.ply', '.pcd']:
        try:
            pcd = o3d.io.read_point_cloud(str(filepath))
            if voxel_size is not None:
                pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
                print(f"  Voxel downsampled to {len(pcd.points)} points (voxel_size={voxel_size})")
            points = np.asarray(pcd.points)
            if load_colors and pcd.has_colors():
                colors = np.asarray(pcd.colors) * 255.0  # Convert to [0, 255]
            if load_normals and pcd.has_normals():
                normals = np.asarray(pcd.normals)
            elif load_normals and not pcd.has_normals():
                # Estimate normals if requested but not available
                pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=0.02, max_nn=30))
                normals = np.asarray(pcd.normals)
        except ImportError:
            raise ImportError("open3d is required for .ply and .pcd files. Install with: pip install open3d")
    elif filepath.suffix == '.xyz':
        data = np.loadtxt(filepath)
        points = data[:, :3]
        if load_colors and data.shape[1] >= 6:
            colors = data[:, 3:6]
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    # Random downsampling (applied to all formats)
    if max_points is not None and len(points) > max_points:
        rng = np.random.default_rng(seed=42)
        indices = rng.choice(len(points), size=max_points, replace=False)
        points = points[indices]
        if colors is not None:
            colors = colors[indices]
        if normals is not None:
            normals = normals[indices]
        print(f"  Randomly downsampled to {max_points} points")

    return PointCloud(points, colors, normals)


def chamfer_distance(pc1: np.ndarray, pc2: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute Chamfer Distance between two point clouds.
    
    Args:
        pc1: Point cloud 1, shape (N, 3)
        pc2: Point cloud 2, shape (M, 3)
    
    Returns:
        Tuple of (chamfer_distance, pc1_to_pc2, pc2_to_pc1)
    """
    from scipy.spatial import cKDTree
    
    # Build KD-trees for efficient nearest neighbor search
    tree1 = cKDTree(pc1)
    tree2 = cKDTree(pc2)
    
    # For each point in pc1, find distance to nearest point in pc2
    distances_1_to_2, _ = tree2.query(pc1, k=1)
    # For each point in pc2, find distance to nearest point in pc1
    distances_2_to_1, _ = tree1.query(pc2, k=1)
    
    # Chamfer distance is the mean of both directions
    cd_1_to_2 = np.mean(distances_1_to_2)
    cd_2_to_1 = np.mean(distances_2_to_1)
    chamfer_dist = (cd_1_to_2 + cd_2_to_1) / 2.0
    
    return chamfer_dist, cd_1_to_2, cd_2_to_1


def hausdorff_distance(pc1: np.ndarray, pc2: np.ndarray) -> float:
    """
    Compute Hausdorff Distance between two point clouds.
    
    Args:
        pc1: Point cloud 1, shape (N, 3)
        pc2: Point cloud 2, shape (M, 3)
    
    Returns:
        Hausdorff distance
    """

    
    tree1 = cKDTree(pc1)
    tree2 = cKDTree(pc2)
    
    distances_1_to_2, _ = tree2.query(pc1, k=1)
    distances_2_to_1, _ = tree1.query(pc2, k=1)
    
    # Hausdorff distance is the maximum of the minimum distances
    hausdorff_dist = max(np.max(distances_1_to_2), np.max(distances_2_to_1))
    
    return hausdorff_dist


def precision_recall_fscore(pc1: np.ndarray, pc2: np.ndarray, threshold: float = 0.01) -> Dict[str, float]:
    """
    Compute Precision, Recall, and F-score at a given threshold.
    
    Args:
        pc1: Point cloud 1 (predicted), shape (N, 3)
        pc2: Point cloud 2 (ground truth), shape (M, 3)
        threshold: Distance threshold for considering a point as correct
    
    Returns:
        Dictionary with precision, recall, and f-score
    """
    from scipy.spatial import cKDTree
    
    tree1 = cKDTree(pc1)
    tree2 = cKDTree(pc2)
    
    # Precision: how many predicted points are close to GT
    distances_1_to_2, _ = tree2.query(pc1, k=1)
    precision = np.mean(distances_1_to_2 < threshold)
    
    # Recall: how many GT points are close to predictions
    distances_2_to_1, _ = tree1.query(pc2, k=1)
    recall = np.mean(distances_2_to_1 < threshold)
    
    # F-score
    if precision + recall > 0:
        fscore = 2 * (precision * recall) / (precision + recall)
    else:
        fscore = 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'fscore': fscore
    }


def rmse(pc1: np.ndarray, pc2: np.ndarray) -> float:
    """
    Compute Root Mean Square Error between corresponding points.
    Assumes point clouds are aligned and have the same number of points.
    
    Args:
        pc1: Point cloud 1, shape (N, 3)
        pc2: Point cloud 2, shape (N, 3)
    
    Returns:
        RMSE value
    """
    if pc1.shape != pc2.shape:
        print("Warning: Point clouds have different shapes. Computing nearest neighbor RMSE.")
        from scipy.spatial import cKDTree
        tree2 = cKDTree(pc2)
        distances, _ = tree2.query(pc1, k=1)
        return np.sqrt(np.mean(distances ** 2))
    
    return np.sqrt(np.mean(np.sum((pc1 - pc2) ** 2, axis=1)))


def rgb_to_ycbcr(rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB to YCbCr color space.
    
    Args:
        rgb: RGB values in [0, 255], shape (N, 3)
    
    Returns:
        YCbCr values, shape (N, 3)
    """
    rgb = rgb.astype(np.float32)
    ycbcr = np.zeros_like(rgb)
    
    ycbcr[:, 0] = 0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]
    ycbcr[:, 1] = 128 - 0.168736 * rgb[:, 0] - 0.331264 * rgb[:, 1] + 0.5 * rgb[:, 2]
    ycbcr[:, 2] = 128 + 0.5 * rgb[:, 0] - 0.418688 * rgb[:, 1] - 0.081312 * rgb[:, 2]
    
    return ycbcr


def psnr_point_to_plane_d2(pc1: PointCloud, pc2: PointCloud, bit_depth: int = 10) -> float:
    """
    Compute Point-to-Plane PSNR MSE D2 metric.
    
    The error is projected along the surface normal of the nearest neighbor.
    
    Args:
        pc1: Reference point cloud with normals
        pc2: Distorted point cloud
        bit_depth: Bit depth for peak calculation (default: 10)
    
    Returns:
        PSNR MSE D2 value
    """
    if pc1.normals is None:
        raise ValueError("Reference point cloud must have normals for D2 metric")
    
    tree1 = cKDTree(pc1.points)
    
    # For each point in pc2, find nearest neighbor in pc1
    distances, indices = tree1.query(pc2.points, k=1)
    
    # Compute error vectors
    error_vectors = pc2.points - pc1.points[indices]
    
    # Project errors along normals
    normals = pc1.normals[indices]
    projected_errors = np.abs(np.sum(error_vectors * normals, axis=1))
    
    # Compute MSE
    mse = np.mean(projected_errors ** 2)
    
    # Compute PSNR
    peak = 2 ** bit_depth - 1
    if mse == 0:
        return float('inf')
    
    psnr = 10 * np.log10((3 * peak ** 2) / mse)
    
    return psnr


def psnr_point_to_attribute_yuv(pc1: PointCloud, pc2: PointCloud) -> Dict[str, float]:
    """
    Compute Point-to-Attribute PSNR MSE YUV metric.
    
    Computes PSNR for color attributes in YCbCr space.
    
    Args:
        pc1: Reference point cloud with colors
        pc2: Distorted point cloud with colors
    
    Returns:
        Dictionary with PSNR values for Y, Cb, Cr, and overall color PSNR
    """
    if pc1.colors is None or pc2.colors is None:
        raise ValueError("Both point clouds must have color information")
    
    # Convert colors to YCbCr
    ycbcr1 = rgb_to_ycbcr(pc1.colors)
    ycbcr2 = rgb_to_ycbcr(pc2.colors)
    
    tree1 = cKDTree(pc1.points)
    tree2 = cKDTree(pc2.points)
    
    # Compute symmetric distances for each channel
    psnr_values = {}
    peak = 255.0
    
    for i, channel in enumerate(['Y', 'Cb', 'Cr']):
        # pc1 to pc2
        _, indices_1to2 = tree2.query(pc1.points, k=1)
        mse_1to2 = np.mean((ycbcr1[:, i] - ycbcr2[indices_1to2, i]) ** 2)
        psnr_1to2 = 10 * np.log10(peak ** 2 / mse_1to2) if mse_1to2 > 0 else float('inf')
        
        # pc2 to pc1
        _, indices_2to1 = tree1.query(pc2.points, k=1)
        mse_2to1 = np.mean((ycbcr2[:, i] - ycbcr1[indices_2to1, i]) ** 2)
        psnr_2to1 = 10 * np.log10(peak ** 2 / mse_2to1) if mse_2to1 > 0 else float('inf')
        
        # Take maximum (symmetric)
        psnr_values[f'psnr_{channel.lower()}'] = max(psnr_1to2, psnr_2to1)
    
    # Compute overall PSNR Color
    psnr_color = (6 * psnr_values['psnr_y'] + psnr_values['psnr_cb'] + 
                  psnr_values['psnr_cr']) / 8
    psnr_values['psnr_color'] = psnr_color
    
    return psnr_values


def pointssim(pc1: PointCloud, pc2: PointCloud, k_neighbors: int = 12,
              use_geometry: bool = True, use_color: bool = True) -> float:
    """
    Compute Point Cloud Structural Similarity (PointSSIM) metric.
    
    Extracts features based on local topology and appearance.
    
    Args:
        pc1: Reference point cloud
        pc2: Distorted point cloud
        k_neighbors: Number of nearest neighbors for feature extraction
        use_geometry: Whether to use geometry features
        use_color: Whether to use color features
    
    Returns:
        PointSSIM score (lower is better, 0 means identical)
    """
    tree1 = cKDTree(pc1.points)
    tree2 = cKDTree(pc2.points)
    
    # Find correspondences
    _, indices = tree1.query(pc2.points, k=1)
    
    similarity_scores = []
    epsilon = 1e-10
    
    for i, point in enumerate(pc2.points):
        ref_idx = indices[i]
        ref_point = pc1.points[ref_idx]
        
        # Get k nearest neighbors
        _, neighbors_ref = tree1.query(ref_point, k=k_neighbors + 1)
        _, neighbors_dist = tree2.query(point, k=k_neighbors + 1)
        
        features_ref = []
        features_dist = []
        
        if use_geometry:
            # Compute geometric features (variance of distances)
            dists_ref = np.linalg.norm(pc1.points[neighbors_ref[1:]] - ref_point, axis=1)
            dists_dist = np.linalg.norm(pc2.points[neighbors_dist[1:]] - point, axis=1)
            
            var_ref = np.var(dists_ref)
            var_dist = np.var(dists_dist)
            
            features_ref.append(var_ref)
            features_dist.append(var_dist)
        
        if use_color and pc1.colors is not None and pc2.colors is not None:
            # Compute color features (variance of colors)
            colors_ref = pc1.colors[neighbors_ref[1:]]
            colors_dist = pc2.colors[neighbors_dist[1:]]
            
            for c in range(3):
                var_ref = np.var(colors_ref[:, c])
                var_dist = np.var(colors_dist[:, c])
                features_ref.append(var_ref)
                features_dist.append(var_dist)
        
        # Compute similarity as relative difference
        if len(features_ref) > 0:
            features_ref = np.array(features_ref)
            features_dist = np.array(features_dist)
            
            similarity = np.abs(features_ref - features_dist) / (
                np.maximum(np.abs(features_ref), np.abs(features_dist)) + epsilon
            )
            similarity_scores.append(np.mean(similarity))
    
    # Average across all points
    pointssim_score = np.mean(similarity_scores)
    
    return pointssim_score


def compute_curvature(points: np.ndarray, neighbors_indices: np.ndarray) -> np.ndarray:
    """
    Compute curvature for points based on their neighbors.
    
    Args:
        points: Point cloud coordinates (N, 3)
        neighbors_indices: Indices of neighbors for each point (N, k)
    
    Returns:
        Curvature values (N,)
    """
    curvatures = []
    
    for i, point in enumerate(points):
        neighbors = points[neighbors_indices[i]]
        
        # Compute covariance matrix
        centered = neighbors - point
        cov = np.cov(centered.T)
        
        # Eigenvalues give principal curvatures
        eigenvalues = np.linalg.eigvalsh(cov)
        
        # Use ratio of smallest to largest eigenvalue as curvature measure
        if eigenvalues[-1] > 1e-10:
            curvature = eigenvalues[0] / eigenvalues[-1]
        else:
            curvature = 0.0
        
        curvatures.append(curvature)
    
    return np.array(curvatures)


def rgb_to_lab2000hl(rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB to LAB2000HL perceptual color space (simplified LAB conversion).
    
    Args:
        rgb: RGB values in [0, 255], shape (N, 3)
    
    Returns:
        LAB values, shape (N, 3)
    """
    # Simplified conversion: normalize RGB and use basic LAB-like transform
    rgb = rgb / 255.0
    
    # Apply gamma correction
    rgb = np.where(rgb > 0.04045,
                   np.power((rgb + 0.055) / 1.055, 2.4),
                   rgb / 12.92)
    
    # Convert to XYZ
    xyz = np.zeros_like(rgb)
    xyz[:, 0] = rgb[:, 0] * 0.4124 + rgb[:, 1] * 0.3576 + rgb[:, 2] * 0.1805
    xyz[:, 1] = rgb[:, 0] * 0.2126 + rgb[:, 1] * 0.7152 + rgb[:, 2] * 0.0722
    xyz[:, 2] = rgb[:, 0] * 0.0193 + rgb[:, 1] * 0.1192 + rgb[:, 2] * 0.9505
    
    # Convert to LAB
    xyz = xyz / np.array([0.95047, 1.00000, 1.08883])
    xyz = np.where(xyz > 0.008856,
                   np.power(xyz, 1/3),
                   (7.787 * xyz) + (16/116))
    
    lab = np.zeros_like(rgb)
    lab[:, 0] = (116 * xyz[:, 1]) - 16  # L
    lab[:, 1] = 500 * (xyz[:, 0] - xyz[:, 1])  # a
    lab[:, 2] = 200 * (xyz[:, 1] - xyz[:, 2])  # b
    
    return lab

def _curvature(pts: np.ndarray, center: np.ndarray) -> float:
    """Smallest/largest eigenvalue ratio of local covariance as curvature proxy."""
    centered = pts - center
    if len(centered) < 3:
        return 0.0
    cov = np.cov(centered.T)
    ev = np.linalg.eigvalsh(cov)
    return float(ev[0] / ev[-1]) if ev[-1] > 1e-10 else 0.0

def pcqm(pc1: PointCloud, pc2: PointCloud, k_neighbors: int = 12) -> float:
    """
    Compute PCQM (Point Cloud Quality Metric).
    
    Implements a data-driven approach with geometry and color features.
    
    Args:
        pc1: Reference point cloud
        pc2: Distorted point cloud
        k_neighbors: Number of nearest neighbors
    
    Returns:
        PCQM score (higher is better, range [0, 1])
    """
    if pc1.colors is None or pc2.colors is None:
        warnings.warn("PCQM requires color information, using geometry only")

    tree1 = cKDTree(pc1.points)
    tree2 = cKDTree(pc2.points)

    # Find correspondences
    _, indices = tree1.query(pc2.points, k=1)

    # Compute features
    f3_values = []  # Curvature structure
    f4_values = []  # Lightness comparison
    f6_values = []  # Lightness structure

    k3, k4, k6 = 1e-4, 1e-2, 1e-2



    for i, point in enumerate(pc2.points):
        ref_idx = indices[i]
        ref_point = pc1.points[ref_idx]

        # Get neighbors
        _, neighbors_ref = tree1.query(ref_point, k=k_neighbors + 1)
        _, neighbors_dist = tree2.query(point, k=k_neighbors + 1)

        # Inline curvature for each point
        curv_ref = _curvature(pc1.points[neighbors_ref], ref_point)
        curv_dist = _curvature(pc2.points[neighbors_dist], point)

        # f3: SSIM-like curvature structure similarity — bounded in (0, 1]
        f3 = (2.0 * curv_ref * curv_dist + k3) / (curv_ref ** 2 + curv_dist ** 2 + k3)
        f3_values.append(f3)

        # Lightness features (if colors available)
        if pc1.colors is not None and pc2.colors is not None:
            lab_ref = rgb_to_lab2000hl(pc1.colors[neighbors_ref].reshape(-1, 3))
            lab_dist = rgb_to_lab2000hl(pc2.colors[neighbors_dist].reshape(-1, 3))

            L_ref = lab_ref[:, 0]
            L_dist = lab_dist[:, 0]

            mu_ref = np.mean(L_ref)
            mu_dist = np.mean(L_dist)
            sigma_L_ref = np.std(L_ref)
            sigma_L_dist = np.std(L_dist)

            # f4: Lightness comparison — bounded in (0, 1]
            f4 = 1.0 / (k4 * (mu_ref - mu_dist) ** 2 + 1.0)
            f4_values.append(f4)

            # f6: Lightness structure (Pearson-like) — bounded in [-1, 1]
            cov_L = np.mean((L_ref - mu_ref) * (L_dist - mu_dist))
            f6 = (cov_L + k6) / (sigma_L_ref * sigma_L_dist + k6)
            f6_values.append(f6)

    # Global features by averaging
    f3_global = float(np.mean(f3_values)) if f3_values else 0.0
    f4_global = float(np.mean(f4_values)) if f4_values else 0.0
    f6_global = float(np.mean(f6_values)) if f6_values else 0.0

    # PCQM score with recommended weights, clamped to [0, 1]
    pcqm_score = 0.18 * f3_global + 0.44 * f4_global + 0.38 * f6_global

    return float(np.clip(pcqm_score, 0.0, 1.0))


def ms_graphsim(pc1: PointCloud, pc2: PointCloud, num_scales: int = 3,
                k_neighbors: int = 20) -> float:
    """
    Compute Multi-Scale Graph Similarity (MS-GraphSIM) metric.
    
    Extends GraphSIM by computing graph-based features at different scales.
    
    Args:
        pc1: Reference point cloud with colors
        pc2: Distorted point cloud with colors
        num_scales: Number of scales to use
        k_neighbors: Number of nearest neighbors
    
    Returns:
        MS-GraphSIM score (higher is better, range [0, 1])
    """
    if pc1.colors is None or pc2.colors is None:
        raise ValueError("MS-GraphSIM requires color information")
    
    # Convert colors to YCbCr-like space for gradient computation
    colors1 = rgb_to_ycbcr(pc1.colors)
    colors2 = rgb_to_ycbcr(pc2.colors)
    
    tree1 = cKDTree(pc1.points)
    tree2 = cKDTree(pc2.points)
    
    # Sample keypoints (use a subset of points)
    n_keypoints = min(1000, len(pc1.points) // 10)
    keypoint_indices = np.random.choice(len(pc1.points), n_keypoints, replace=False)
    
    scale_scores = []
    T0, T1, T2 = 0.001, 0.001, 0.001
    channel_weights = np.array([6.0, 1.0, 1.0]) / 8.0  # Y:Cb:Cr = 6:1:1

    for scale in range(num_scales):
        k = max(2, k_neighbors // (scale + 1))  # Reduce neighbors at higher scales
        similarities = []

        for idx in keypoint_indices:
            keypoint = pc1.points[idx]

            # Find neighbors at current scale
            _, neighbors_ref = tree1.query(keypoint, k=k + 1)

            # Find corresponding point in distorted cloud
            _, dist_idx = tree2.query(keypoint, k=1)
            _, neighbors_dist = tree2.query(pc2.points[dist_idx], k=k + 1)

            colors_ref_nb = colors1[neighbors_ref[1:]]   # (k, 3)
            colors_dist_nb = colors2[neighbors_dist[1:]] # (k, 3)

            # Compute per-channel SSIM-like components
            per_channel_sim = []
            for c in range(3):
                cr = colors_ref_nb[:, c]          # neighbor values, ref
                cd = colors_dist_nb[:, c]         # neighbor values, dist
                ref_c = colors1[idx, c]           # centre node, ref
                dist_c = colors2[dist_idx, c]     # centre node, dist

                # Gradient mass: total absolute deviation from centre node
                # Formula: (2*mg_r*mg_d + C) / (mg_r^2 + mg_d^2 + C) ∈ (0, 1]
                mg_r = np.sum(np.abs(cr - ref_c))
                mg_d = np.sum(np.abs(cd - dist_c))
                sim_mg = (2.0 * mg_r * mg_d + T0) / (mg_r ** 2 + mg_d ** 2 + T0)

                # Gradient mean: (2*mu_r*mu_d + C) / (mu_r^2 + mu_d^2 + C) ∈ (0, 1]
                mu_r = np.mean(cr)
                mu_d = np.mean(cd)
                sim_mu = (2.0 * mu_r * mu_d + T1) / (mu_r ** 2 + mu_d ** 2 + T1)

                # Gradient covariance structure (Pearson-like) ∈ [-1, 1]
                sig_r = np.std(cr)
                sig_d = np.std(cd)
                cov_val = np.mean((cr - mu_r) * (cd - mu_d))
                sim_cg = (cov_val + T2) / (sig_r * sig_d + T2)

                per_channel_sim.append(sim_mg * sim_mu * sim_cg)

            # Weighted combination across channels: 6:1:1 for Y:Cb:Cr
            point_sim = float(np.dot(channel_weights, per_channel_sim))
            similarities.append(point_sim)

        scale_scores.append(np.mean(similarities))

    # Aggregate across scales and clamp to [0, 1]
    ms_graphsim_score = float(np.clip(np.mean(scale_scores), 0.0, 1.0))

    return ms_graphsim_score


def compute_all_metrics(pc1: PointCloud, pc2: PointCloud, threshold: float = 0.01,
                       include_advanced: bool = True) -> Dict[str, float]:
    """
    Compute all metrics between two point clouds.
    
    Args:
        pc1: Point cloud 1 (reference)
        pc2: Point cloud 2 (distorted)
        threshold: Distance threshold for precision/recall
        include_advanced: Whether to compute advanced metrics (PointSSIM, PCQM, etc.)
    
    Returns:
        Dictionary with all computed metrics
    """
    metrics = {}
    
    print("Computing Chamfer Distance...")
    cd, cd_1to2, cd_2to1 = chamfer_distance(pc1.points, pc2.points)
    metrics['chamfer_distance'] = cd
    metrics['chamfer_pc1_to_pc2'] = cd_1to2
    metrics['chamfer_pc2_to_pc1'] = cd_2to1
    
    print("Computing Hausdorff Distance...")
    hd = hausdorff_distance(pc1.points, pc2.points)
    metrics['hausdorff_distance'] = hd
    
    print("Computing Precision, Recall, F-score...")
    prf = precision_recall_fscore(pc1.points, pc2.points, threshold)
    metrics.update(prf)
    
    print("Computing RMSE...")
    rmse_val = rmse(pc1.points, pc2.points)
    metrics['rmse'] = rmse_val
    
    if include_advanced:
        # Point-to-Plane D2 (requires normals)
        if pc1.normals is not None:
            try:
                print("Computing Point-to-Plane PSNR MSE D2...")
                psnr_d2 = psnr_point_to_plane_d2(pc1, pc2)
                metrics['psnr_d2'] = psnr_d2
            except Exception as e:
                print(f"Warning: Could not compute PSNR D2: {e}")
        
        # Point-to-Attribute YUV (requires colors)
        if pc1.colors is not None and pc2.colors is not None:
            try:
                print("Computing Point-to-Attribute PSNR MSE YUV...")
                psnr_yuv = psnr_point_to_attribute_yuv(pc1, pc2)
                metrics.update(psnr_yuv)
            except Exception as e:
                print(f"Warning: Could not compute PSNR YUV: {e}")
            
            try:
                print("Computing PointSSIM (Geometry)...")
                pssim_geom = pointssim(pc1, pc2, use_geometry=True, use_color=False)
                metrics['pointssim_geometry'] = pssim_geom
            except Exception as e:
                print(f"Warning: Could not compute PointSSIM Geometry: {e}")
            
            try:
                print("Computing PointSSIM (Luminance)...")
                pssim_color = pointssim(pc1, pc2, use_geometry=False, use_color=True)
                metrics['pointssim_luminance'] = pssim_color
            except Exception as e:
                print(f"Warning: Could not compute PointSSIM Luminance: {e}")
            
            try:
                print("Computing PCQM...")
                pcqm_score = pcqm(pc1, pc2)
                metrics['pcqm'] = pcqm_score
            except Exception as e:
                print(f"Warning: Could not compute PCQM: {e}")
            
            try:
                print("Computing MS-GraphSIM...")
                ms_graph_score = ms_graphsim(pc1, pc2)
                metrics['ms_graphsim'] = ms_graph_score
            except Exception as e:
                print(f"Warning: Could not compute MS-GraphSIM: {e}")
    
    return metrics


def visualize_point_clouds(pc1: PointCloud, pc2: PointCloud):
    """
    Visualize two point clouds side by side using Open3D.
    
    Args:
        pc1: Point cloud 1
        pc2: Point cloud 2
    """
    try:
        import open3d as o3d
    except ImportError:
        print("Open3D not installed. Skipping visualization.")
        print("Install with: pip install open3d")
        return
    
    # Create Open3D point cloud objects
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pc1.points)
    if pc1.colors is not None:
        pcd1.colors = o3d.utility.Vector3dVector(pc1.colors / 255.0)
    else:
        pcd1.paint_uniform_color([1, 0, 0])  # Red
    
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(pc2.points)
    if pc2.colors is not None:
        pcd2.colors = o3d.utility.Vector3dVector(pc2.colors / 255.0)
    else:
        pcd2.paint_uniform_color([0, 0, 1])  # Blue
    
    # Visualize
    print("\nVisualizing point clouds...")
    print("Left: Point Cloud 1 (Reference)")
    print("Right: Point Cloud 2 (Distorted)")
    o3d.visualization.draw_geometries([pcd1, pcd2],
                                      window_name="Point Cloud Comparison",
                                      width=1024, height=768)


def display_metrics(metrics: Dict[str, float]):
    """
    Display metrics in a formatted table.
    
    Args:
        metrics: Dictionary of metric names and values
    """
    print("\n" + "="*70)
    print("POINT CLOUD COMPARISON METRICS")
    print("="*70)
    
    # Basic geometric metrics
    print("\n[BASIC GEOMETRIC METRICS]")
    if 'chamfer_distance' in metrics:
        print(f"Chamfer Distance (CD):")
        print(f"  Overall CD:        {metrics['chamfer_distance']:.6f}")
        print(f"  PC1 → PC2:         {metrics['chamfer_pc1_to_pc2']:.6f}")
        print(f"  PC2 → PC1:         {metrics['chamfer_pc2_to_pc1']:.6f}")
    
    if 'hausdorff_distance' in metrics:
        print(f"\nHausdorff Distance:  {metrics['hausdorff_distance']:.6f}")
    
    if 'rmse' in metrics:
        print(f"RMSE:                {metrics['rmse']:.6f}")
    
    if 'precision' in metrics:
        print(f"\nPrecision/Recall Metrics:")
        print(f"  Precision:         {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
        print(f"  Recall:            {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
        print(f"  F-score:           {metrics['fscore']:.4f}")
    
    # Advanced metrics
    has_advanced = any(k in metrics for k in ['psnr_d2', 'psnr_color', 'pointssim_geometry', 
                                                'pointssim_luminance', 'pcqm', 'ms_graphsim'])
    
    if has_advanced:
        print("\n" + "="*70)
        print("[ADVANCED QUALITY METRICS]")
        
        if 'psnr_d2' in metrics:
            print(f"\nPSNR MSE D2 (Point-to-Plane):")
            print(f"  PSNR D2:           {metrics['psnr_d2']:.4f} dB")
        
        if 'psnr_color' in metrics:
            print(f"\nPSNR MSE YUV (Point-to-Attribute):")
            if 'psnr_y' in metrics:
                print(f"  PSNR Y:            {metrics['psnr_y']:.4f} dB")
            if 'psnr_cb' in metrics:
                print(f"  PSNR Cb:           {metrics['psnr_cb']:.4f} dB")
            if 'psnr_cr' in metrics:
                print(f"  PSNR Cr:           {metrics['psnr_cr']:.4f} dB")
            print(f"  PSNR Color:        {metrics['psnr_color']:.4f} dB")
        
        if 'pointssim_geometry' in metrics or 'pointssim_luminance' in metrics:
            print(f"\nPointSSIM (Structural Similarity):")
            if 'pointssim_geometry' in metrics:
                print(f"  Geometry Features: {metrics['pointssim_geometry']:.6f}")
            if 'pointssim_luminance' in metrics:
                print(f"  Luminance Features:{metrics['pointssim_luminance']:.6f}")
            print(f"  (Lower is better, 0 = identical)")
        
        if 'pcqm' in metrics:
            print(f"\nPCQM (Point Cloud Quality Metric):")
            print(f"  PCQM Score:        {metrics['pcqm']:.6f}")
            print(f"  (Higher is better, range [0, 1])")
        
        if 'ms_graphsim' in metrics:
            print(f"\nMS-GraphSIM (Multi-Scale Graph Similarity):")
            print(f"  MS-GraphSIM:       {metrics['ms_graphsim']:.6f}")
            print(f"  (Higher is better, range [0, 1])")
    
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compare two point clouds and compute various quality metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic metrics only
  python pointcloud_metrics.py reference.ply distorted.ply

  # With visualization
  python pointcloud_metrics.py reference.ply distorted.ply --visualize

  # Include advanced metrics (requires colors/normals)
  python pointcloud_metrics.py reference.ply distorted.ply --advanced

  # Compute normals for PSNR D2 metric
  python pointcloud_metrics.py reference.ply distorted.ply --normals --advanced

  # Save results to JSON
  python pointcloud_metrics.py ref.ply dist.ply --output results.json

Available Metrics:
  Basic:
    - Chamfer Distance (CD)
    - Hausdorff Distance
    - Precision, Recall, F-score
    - RMSE
  
  Advanced (requires --advanced flag):
    - PSNR MSE D2 (Point-to-Plane, requires normals)
    - PSNR MSE YUV (Point-to-Attribute, requires colors)
    - PointSSIM (Structural Similarity, geometry & luminance)
    - PCQM (Point Cloud Quality Metric)
    - MS-GraphSIM (Multi-Scale Graph Similarity)
        """
    )
    parser.add_argument("pc1", type=str, help="Path to reference point cloud file")
    parser.add_argument("pc2", type=str, help="Path to distorted point cloud file")
    parser.add_argument("--threshold", type=float, default=0.01,
                        help="Distance threshold for precision/recall (default: 0.01)")
    parser.add_argument("--advanced", action="store_true",
                        help="Compute advanced quality metrics (PSNR, PointSSIM, PCQM, MS-GraphSIM)")
    parser.add_argument("--normals", action="store_true",
                        help="Compute/load normals for Point-to-Plane D2 metric")
    parser.add_argument("--voxel-size", type=float, default=None,
                        help="Voxel size for voxel grid downsampling (only .ply/.pcd)")
    parser.add_argument("--max-points", type=int, default=None,
                        help="Maximum number of points after random downsampling")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize the point clouds using Open3D")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file to save metrics (JSON format)")
    
    args = parser.parse_args()
    
    print(f"Loading reference point cloud from: {args.pc1}")
    pc1 = load_point_cloud(args.pc1, load_colors=True, load_normals=args.normals,
                           voxel_size=args.voxel_size, max_points=args.max_points)
    print(f"  Points: {len(pc1)}")
    print(f"  Has colors: {pc1.colors is not None}")
    print(f"  Has normals: {pc1.normals is not None}")
    
    print(f"\nLoading distorted point cloud from: {args.pc2}")
    pc2 = load_point_cloud(args.pc2, load_colors=True, load_normals=False,
                           voxel_size=args.voxel_size, max_points=args.max_points)
    print(f"  Points: {len(pc2)}")
    print(f"  Has colors: {pc2.colors is not None}")
    print(f"  Has normals: {pc2.normals is not None}")
    
    # Compute metrics
    print("\n" + "="*70)
    metrics = compute_all_metrics(pc1, pc2, args.threshold, 
                                  include_advanced=args.advanced)
    
    # Display metrics
    display_metrics(metrics)
    
    # Save metrics to file if requested
    if args.output:
        import json
        output_path = Path(args.output)
        # Convert numpy scalars to native floats and inf values to a string for JSON serialization
        metrics_serializable = {}
        for k, v in metrics.items():
            v = float(v)
            metrics_serializable[k] = "inf" if np.isinf(v) else v
        
        with open(output_path, 'w') as f:
            json.dump(metrics_serializable, f, indent=2)
        print(f"Metrics saved to: {output_path}")
    
    # Visualize if requested
    if args.visualize:
        visualize_point_clouds(pc1, pc2)


if __name__ == "__main__":
    main()
