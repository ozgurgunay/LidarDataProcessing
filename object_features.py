import numpy as np

def extract_features(points, labels):
    """
    Extracts key features for each object cluster, such as its dimensions,
    point count, and average intensity.

    Args:
        points (np.ndarray): The non-ground points (N x 4 array: X, Y, Z, Intensity).
        labels (np.ndarray): The cluster labels from DBSCAN for each point.

    Returns:
        List[dict]: A list of dictionaries, where each dictionary contains the
                    features of a single object cluster.
    """
    features = []
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        # Skip noise points, which are labeled as -1
        if label == -1:
            continue

        # Get all points belonging to the current cluster
        cluster_points = points[labels == label]

        # --- NEW FEATURE: Average Intensity ---
        # Calculate the average reflection intensity of the points in the cluster (from the 4th column)
        avg_intensity = int(np.mean(cluster_points[:, 3]))

        # Calculate the spatial dimensions of the cluster using only XYZ coordinates
        min_bound = np.min(cluster_points[:, :3], axis=0)
        max_bound = np.max(cluster_points[:, :3], axis=0)
        extent = max_bound - min_bound  # This gives [width, length, height]

        feature = {
            "label": int(label),
            "num_points": int(len(cluster_points)),
            "width": round(float(extent[0]), 2),
            "length": round(float(extent[1]), 2),
            "height": round(float(extent[2]), 2),
            "avg_intensity": avg_intensity,
            "bbox_min": [round(float(x), 2) for x in min_bound],
            "bbox_max": [round(float(x), 2) for x in max_bound]
        }
        features.append(feature)
        
    return features


def classify_object_advanced(feature: dict) -> str:
    """
    Classifies objects based on their features using a set of flexible, heuristic rules.
    
    Args:
        feature (dict): A dictionary of features created by the extract_features function.
        
    Returns:
        str: The class label for the object ("car", "pedestrian", "cyclist", "noise", or "unknown").
    """
    n_points = feature["num_points"]
    
    # To make the classification independent of the object's orientation,
    # we define 'width' as the smaller dimension and 'length' as the larger one.
    dims = sorted([feature["width"], feature["length"]])
    width, length = dims[0], dims[1]
    height = feature["height"]
    
    # --- RULE 1: Noise Filter ---
    # Clusters with very few points or that are extremely small are likely noise.
    if n_points < 20 or (length < 0.2 and width < 0.2 and height < 0.2):
        return "noise"

    # --- RULE 2: Pedestrian or Cyclist ---
    # These objects are typically 'vertical' and 'narrow'.
    # Their height is significantly greater than their width and length.
    if height > 1.0 and length < 1.5 and width < 1.5:
        # A simple check to distinguish between pedestrians and cyclists
        if length > 1.0:  # Cyclists are generally longer than pedestrians
            return "cyclist"
        else:
            return "pedestrian"

    # --- RULE 3: Car ---
    # Cars are typically 'horizontal' and 'large'.
    # They are larger than a certain size but not excessively tall.
    if length > 1.5 and width > 1.0 and height > 0.8 and height < 3.0:
        return "car"
    
    # --- RULE 4: Everything Else ---
    # Any cluster that doesn't fit the rules above is labeled as 'unknown'.
    return "unknown"