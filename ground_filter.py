import numpy as np
import open3d as o3d

"""
ground_filter.py

This file contains the function used to filter out ground points from the point cloud.
"""

def filter_ground_ransac(points, distance_threshold=0.2, ransac_n=3, num_iterations=1000):
    """
    Finds the ground plane using the RANSAC algorithm and returns the non-ground points.
    This method works well even on sloped surfaces and preserves all feature columns (e.g., intensity).

    Args:
        points (np.ndarray): The input point cloud as an array (N x features, e.g., X,Y,Z,Intensity).
        distance_threshold (float): The maximum distance a point can be from the plane to be
                                    considered part of the ground.
        ransac_n (int): The number of points used to estimate the plane model.
        num_iterations (int): The number of iterations the RANSAC algorithm runs.

    Returns:
        np.ndarray: An array containing only the non-ground points.
    """
    if points is None or points.shape[0] < ransac_n:
        print("[filter_ground_ransac] Invalid or insufficient number of points.")
        return np.empty((0, points.shape[1]))

    # Create an Open3D PointCloud object, using only the XYZ coordinates for plane fitting.
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])

    # Segment the largest plane, which we assume is the ground.
    _, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations
    )
    
    # The 'inliers' are the indices of the ground points. We want everything else.
    # Let's remove the ground points from the original array to get the non-ground points.
    non_ground_points = np.delete(points, inliers, axis=0)
    
    print(f"Number of non-ground points after RANSAC: {non_ground_points.shape[0]}")
    
    return non_ground_points