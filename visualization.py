import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

def visualize_points(original_pts_with_intensity, non_ground_pts_with_intensity):
    """
    Visualizes the point cloud in 3D, showing the ground and non-ground points
    in different colors.

    Args:
        original_pts_with_intensity (np.ndarray): The complete point cloud (N x 4).
        non_ground_pts_with_intensity (np.ndarray): The points classified as non-ground (M x 4).
    """
    print(f"Visualizing points: {original_pts_with_intensity.shape[0]} original, {non_ground_pts_with_intensity.shape[0]} non-ground.")
    
    # We only need the XYZ coordinates for visualization.
    original_xyz = original_pts_with_intensity[:, :3]
    non_ground_xyz = non_ground_pts_with_intensity[:, :3]

    # A fast way to find the ground points is to see which of the original points
    # are NOT in the non_ground set.
    non_ground_set = {tuple(row) for row in np.round(non_ground_xyz, 5)}
    ground_mask = [tuple(row) not in non_ground_set for row in np.round(original_xyz, 5)]
    ground_xyz = original_xyz[ground_mask]

    # Create separate PointCloud objects for ground and non-ground points.
    # This is a clean and efficient way to color them differently.
    pcd_non_ground = o3d.geometry.PointCloud()
    pcd_non_ground.points = o3d.utility.Vector3dVector(non_ground_xyz)
    pcd_non_ground.paint_uniform_color([0, 1, 0])  # Green for non-ground (objects)

    pcd_ground = o3d.geometry.PointCloud()
    pcd_ground.points = o3d.utility.Vector3dVector(ground_xyz)
    pcd_ground.paint_uniform_color([0.5, 0.5, 0.5])  # Gray for ground

    # Display both point clouds together in the same window.
    o3d.visualization.draw_geometries([pcd_ground, pcd_non_ground])

def visualize_clusters(points_with_intensity, labels):
    """
    Visualizes the clustered points in 3D, with each cluster shown in a different color.
    
    Args:
        points_with_intensity (np.ndarray): The non-ground points (N x 4).
        labels (np.ndarray): The cluster label for each point.
    """
    points_xyz = points_with_intensity[:, :3]
    
    unique_labels = np.unique(labels)
    # Use a colormap to assign a unique color to each cluster ID.
    cmap = plt.get_cmap("tab20")
    colors = np.zeros_like(points_xyz)

    for lbl in unique_labels:
        if lbl == -1:
            # Noise points are colored gray.
            colors[labels == lbl] = [0.5, 0.5, 0.5]
        else:
            # Assign a color from the colormap based on the cluster label.
            rgb = cmap(lbl % 20)[:3]
            colors[labels == lbl] = rgb

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd])

def visualize_tracked_objects(points_with_intensity, features):
    """
    Draws bounding boxes and their tracking IDs over the detected objects.

    Args:
        points_with_intensity (np.ndarray): The non-ground points (N x 4).
        features (list): A list of dictionaries, each containing the features and
                         the 'object_id' for a detected object.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_with_intensity[:, :3])
    # Start by coloring all points gray as a background.
    pcd.paint_uniform_color([0.5, 0.5, 0.5])

    geometries = [pcd]
    cmap = plt.get_cmap("tab20")

    for feat in features:
        obj_id = feat.get("object_id", -1)
        # Only draw boxes for objects that have been assigned an ID and are not noise.
        if obj_id != -1 and feat.get("class") != "noise":
            min_bound = feat["bbox_min"]
            max_bound = feat["bbox_max"]
            
            # Create an axis-aligned bounding box for Open3D.
            bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)
            
            # Assign a color to the box based on its tracking ID.
            color = cmap(obj_id % 20)[:3]
            bbox.color = color
            geometries.append(bbox)

    print("\nVisualizing tracked objects with their IDs...")
    o3d.visualization.draw_geometries(geometries)