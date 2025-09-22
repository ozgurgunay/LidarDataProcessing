import os
import pandas as pd
from glob import glob

def get_all_csv_files(data_root):
    """
    Finds the paths of all CSV files within a given root directory.
    Args:
        data_root (str): The main data folder.

    Returns:
        List[str]: A sorted list of full paths to the CSV files.
    """
    pattern = os.path.join(data_root, "**", "*.csv")
    return sorted(glob(pattern, recursive=True))


def read_lidar_csv(path):
    """
    Reads a LiDAR frame from a CSV file and returns the X, Y, Z, and INTENSITY data.
    Args:
        path (str): The path to the CSV file.

    Returns:
        np.ndarray: An array of points with their features (X, Y, Z, Intensity), shape (N, 4).
                    Returns None if the file cannot be read.
    """
    try:
        df = pd.read_csv(path, sep=';', encoding='utf-8')
        # Clean up column names to be consistent (e.g., remove whitespace, make uppercase)
        df.columns = df.columns.str.strip().str.upper()

        # We expect these specific columns to be in the file.
        expected_cols = ['X', 'Y', 'Z', 'INTENSITY']
        if not all(col in df.columns for col in expected_cols):
            raise ValueError(f"ERROR: Missing required columns. Expected {expected_cols}, but found: {df.columns.tolist()}")

        # Extract the required columns and convert them to a NumPy array.
        points_with_intensity = df[expected_cols].values.astype(float)

        if points_with_intensity.shape[0] == 0:
            raise ValueError("The CSV file contains no data points.")

        return points_with_intensity
    
    except Exception as e:
        print(f"ERROR reading file ({path}): {e}")
        return None