# LiDAR Object Detection and Tracking for Autonomous Driving

This project is a complete pipeline for detecting, classifying, and tracking objects from LiDAR point cloud data.

The system processes a sequence of 718 LiDAR frames, identifies objects such as cars, pedestrians, and cyclists, and tracks them across the entire scenario.


---

##  Table of Contents
- [Project Goal](#-project-goal)
- [Core Features](#-core-features)
- [Methodology & Pipeline](#-methodology--pipeline)
- [Technologies Used](#-technologies-used)
- [Setup and Usage](#-setup-and-usage)
- [Results](#-results)

---

##  Project Goal

The main objective of this project is to develop a robust system that enhances situational awareness for autonomous vehicles by processing raw LiDAR data. The system is designed to:
1.  **Detect** potential objects in a 3D environment.
2.  **Classify** them into meaningful categories (Car, Pedestrian, etc.).
3.  **Track** each unique object across a sequence of frames.

---

##  Core Features

- **RANSAC Ground Segmentation:** Reliably separates ground points from potential object points, even on sloped surfaces.
- **DBSCAN Clustering:** Groups non-ground points into distinct object clusters without needing a predefined number of objects.
- **Heuristic-Based Classification:** Classifies objects based on their physical dimensions (size, orientation) using a set of robust rules.
- **Euclidean Distance Tracking:** A simple yet effective tracker that assigns and maintains a unique ID for each object throughout the sequence.
- **Data-Driven Analysis:** Generates detailed JSON outputs for each frame and includes a script to analyze the entire dataset, providing summary statistics and performance metrics.

---

##  Methodology & Pipeline

The algorithm follows a modular, step-by-step pipeline for each LiDAR frame:

1.  **Data Loading:** Reads and parses point cloud data (X, Y, Z, Intensity) from CSV files.
2.  **Ground Removal:** The RANSAC algorithm is applied to identify and remove the dominant ground plane.
3.  **Clustering:** DBSCAN is used to group the remaining non-ground points into object clusters.
4.  **Feature Extraction & Classification:** For each cluster, key features like dimensions and point count are extracted and fed into a rule-based classifier.
5.  **Tracking:** A Euclidean distance tracker associates detections between frames to maintain object identities.
6.  **Output Generation:** The final analysis, including object class and tracking ID, is saved as a JSON file for each frame.

---

##  Technologies Used

- **Language:** Python 3.x
- **Core Libraries:**
  - **NumPy:** For efficient numerical operations and array manipulation.
  - **Pandas:** For reading and parsing the input CSV data.
  - **Open3D:** For advanced 3D data processing, including RANSAC plane segmentation and visualization.
  - **Scikit-learn:** For implementing the DBSCAN clustering algorithm.
  - **Matplotlib:** For generating summary plots and visualizations.

---

##  Results

The system was tested on the full dataset of 718 frames. The analysis revealed the detection and tracking of **[1069]** unique objects throughout the scenario, including **[128]** cars, **[195]** pedestrians, and **[79]** cyclists.

The tracking algorithm proved to be stable, with **[114]** objects being tracked for 10 or more consecutive frames. The longest tracked object, likely a static part of the environment, was followed for **[794]** frames, demonstrating the tracker's robustness.