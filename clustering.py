import numpy as np
from sklearn.cluster import DBSCAN


def cluster_objects(points, eps=0.5, min_samples=10):
    """
    Clusters the non-ground points using the DBSCAN algorithm.

    Args:
        points (np.ndarray): The non-ground points (N x features).
        eps (float): The maximum distance between two samples for one to be considered
                     as in the neighborhood of the other.
        min_samples (int): The number of samples in a neighborhood for a point to be
                           considered as a core point.

    Returns:
        Tuple[np.ndarray, int]: A tuple containing:
            - labels (np.ndarray): Cluster labels for each point. Noise points are given the label -1.
            - n_clusters (int): The total number of clusters found, excluding noise.
    """
    if points.shape[0] == 0:
        return np.array([]), 0

    # Apply DBSCAN only on the XYZ coordinates (the first 3 columns).
    # This prevents the intensity value from affecting the clustering process.
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points[:, :3])
    labels = clustering.labels_

    # Calculate the number of clusters, excluding any noise points.
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    return labels, n_clusters