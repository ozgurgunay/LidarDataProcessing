import os
import json
from collections import Counter
import open3d as o3d

# Import custom modules
from io_utils import get_all_csv_files, read_lidar_csv
from ground_filter import filter_ground_ransac
from clustering import cluster_objects
from object_features import extract_features, classify_object_advanced
from tracker import EuclideanDistTracker
from visualization import visualize_points, visualize_clusters, visualize_tracked_objects

"""
main.py

This is the main script that runs the entire LiDAR processing pipeline.
It calls functions from other modules to perform each step in sequence.
"""

# --- SETUP ---

# Create the output directory if it doesn't exist
os.makedirs("output", exist_ok=True)

# --- CONFIGURATION PARAMETERS ---

# The root folder where the LiDAR data is stored
DATA_ROOT = "data"

# RANSAC parameters for ground filtering
# The max distance (in meters) a point can be from the plane to be considered ground.
# 0.2 meters (20cm) is usually a good starting point.
RANSAC_DISTANCE_THRESHOLD = 0.2

# DBSCAN parameters for object clustering
# These values can be tuned to get the best results.
# eps: Max distance between points to be considered neighbors. Increase for larger objects.
# min_samples: The minimum number of points required to form a cluster.
DBSCAN_EPS = 1.0
DBSCAN_MIN_SAMPLES = 20

# --- INITIALIZATION ---

# Get the paths of all CSV files to be processed
all_paths = get_all_csv_files(DATA_ROOT)
if not all_paths:
    print(f"Warning: No CSV files found in the '{DATA_ROOT}' directory.")
    exit()
print(f"Found {len(all_paths)} frames to process.")

# Set the number of frames to process (use len(all_paths) for the full dataset)
MAX_FRAMES = len(all_paths)

# Initialize the object tracker outside the loop
# This allows the tracker to maintain its memory across all frames.
tracker = EuclideanDistTracker()

# --- PROCESSING LOOP ---

# Process each frame one by one
for i, path in enumerate(all_paths[:MAX_FRAMES]):
    print(f"\n[{i+1}/{MAX_FRAMES}] Processing frame: {os.path.basename(path)}")

    # Step 1: Read the LiDAR data (X, Y, Z, Intensity)
    points_with_intensity = read_lidar_csv(path)
    if points_with_intensity is None:
        print("Skipped: Could not read the file.")
        continue
    print(f"Total points in frame: {points_with_intensity.shape[0]}")

    # Step 2: Filter out the ground using RANSAC
    non_ground_points = filter_ground_ransac(
        points_with_intensity, 
        distance_threshold=RANSAC_DISTANCE_THRESHOLD
    )
    
    if non_ground_points.shape[0] == 0:
        print("Warning: No points left after ground filtering. Skipping frame.")
        continue

    # Step 3: Cluster the remaining points into objects using DBSCAN
    labels, n_clusters = cluster_objects(
        non_ground_points, 
        eps=DBSCAN_EPS, 
        min_samples=DBSCAN_MIN_SAMPLES
    )
    print(f"Found {n_clusters} potential object clusters.")

    # Step 4: Extract features, classify, and track the objects
    if n_clusters > 0:
        features = extract_features(non_ground_points, labels)

        # Prepare a list of detections for the tracker
        detections_for_tracker = []
        for feat in features:
            # Temporarily classify objects to filter out noise before tracking
            feat["class"] = classify_object_advanced(feat)
            if feat["class"] != 'noise':
                # The tracker uses a 2D bounding box (top-down view)
                min_x, min_y, _ = feat["bbox_min"]
                w = feat["width"]
                h = feat["length"]
                detections_for_tracker.append([min_x, min_y, w, h])

        # Update the tracker with the new detections
        tracked_objects = tracker.update(detections_for_tracker)
        
        # Match the tracker IDs back to our feature list
        for feat in features:
            feat['object_id'] = -1  # Default to no ID
            if feat["class"] != 'noise':
                min_x, min_y, _ = feat["bbox_min"]
                # Find the closest tracked object for each feature
                for tobj in tracked_objects:
                    tx, ty, _, _, obj_id = tobj
                    # If the starting points of the bounding boxes are very close, they match.
                    if abs(min_x - tx) < 0.1 and abs(min_y - ty) < 0.1:
                        feat['object_id'] = obj_id
                        break
        
        # Print a summary of the classes and tracked objects for this frame
        class_counts = Counter([f["class"] for f in features])
        print("▶ Class distribution:")
        for cls, count in sorted(class_counts.items()):
            print(f"  - {cls.capitalize()}: {count} clusters")
        print(f"▶ Number of tracked objects: {len(tracked_objects)}")

    else:
        features = []
        # If no objects are detected, update the tracker with an empty list to clear its memory
        tracker.update([])
        print("▶ Class distribution: No clusters found.")

    # Step 5: Visualization (optional, for debugging the first frame)
    # To use this, uncomment the block below and set MAX_FRAMES to a small number.
    # if i == 0:
    #     print("\nOpening visualization windows for the first frame...")
    #     # 1. Show ground vs. non-ground points
    #     visualize_points(points_with_intensity, non_ground_points)
    #     # 2. Show colored clusters
    #     if n_clusters > 0:
    #         visualize_clusters(non_ground_points, labels)
    #         # 3. Show tracked objects with bounding boxes
    #         visualize_tracked_objects(non_ground_points, features)

    # Step 6: Save the results to a JSON file
    # We save the results for every frame for later analysis.
    output_data = [
        {
            "object_id": feat.get("object_id", -1),
            "cluster_id": feat["label"],
            "class": feat["class"],
            "num_points": feat["num_points"],
            "dimensions_m": {
                "width": feat["width"],
                "length": feat["length"],
                "height": feat["height"]
            },
            "avg_intensity": feat.get("avg_intensity", 0)
        }
        for feat in features if feat["class"] != "noise"  # Don't save noise clusters
    ]

    output_filename = f"output/frame_{i+1:03d}_analysis.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    print(f"Analysis results saved to: {output_filename}")

print("\nProcessing complete.")