# LiDAR Object Detection and Tracking for Autonomous Driving

This project implements a complete pipeline for detecting, classifying, and tracking objects using LiDAR point cloud data. The system processes 718 LiDAR frames, identifies objects like cars, pedestrians, and cyclists, and tracks them throughout the entire scenario.

## Table of Contents

1.  Project Goal
2.  Core Features
3.  Methodology & Pipeline
4.  Technologies Used
5.  Results

-----

## 1\. Project Goal

The main goal is to build a strong system that improves situational awareness for autonomous vehicles by processing raw LiDAR data. The system aims to:

  * **Detect** objects in a 3D environment.
  * **Classify** them into groups (e.g., Car, Pedestrian).
  * **Track** each unique object across multiple frames.

-----

## 2\. Core Features

  * **RANSAC Ground Segmentation:** Separates ground points from potential object points, even on sloped surfaces.
  * **DBSCAN Clustering:** Groups non-ground points into distinct object clusters without needing to know the number of objects beforehand.
  * **Heuristic-Based Classification:** Classifies objects based on their size and shape using simple rules.
  * **Euclidean Distance Tracking:** A basic but effective tracker that gives and keeps a unique ID for each object throughout the process.
  * **Data-Driven Analysis:** Creates detailed JSON outputs for each frame and includes a script to analyze the whole dataset, showing summary statistics and performance.

-----

## 3\. Methodology & Pipeline

The algorithm processes each LiDAR frame step-by-step:

1.  **Data Loading:** Reads point cloud data (X, Y, Z, Intensity) from CSV files.
2.  **Ground Removal:** RANSAC finds and removes the main ground plane.
3.  **Clustering:** DBSCAN groups the remaining non-ground points into object clusters.
4.  **Feature Extraction & Classification:** Key features (like size and point count) are taken from each cluster and used by a rule-based classifier.
5.  **Tracking:** A Euclidean distance tracker connects detections between frames to keep object identities consistent.
6.  **Output Generation:** The final analysis, including object class and tracking ID, is saved as a JSON file for each frame.

### Visual Pipeline

Here's how the algorithm's main steps look:
<img width="1920" height="1080" alt="figure4" src="https://github.com/user-attachments/assets/3acae4a4-e816-4204-904a-b4f9a3031c8e" />

*Figure 1: Visualization of Algorithmic Stages. (a) Ground points (gray) are separated from potential object points (green) by RANSAC. (b) DBSCAN groups non-ground points into distinct, color-coded clusters. (c) Each cluster is classified, enclosed in a bounding box, and assigned a tracking ID.*

-----

## 4\. Technologies Used

  * **Language:** Python 3.x
  * **Key Libraries:**
      * **NumPy:** For math operations and array handling.
      * **Pandas:** For reading and managing CSV data.
      * **Open3D:** For advanced 3D data processing, like RANSAC and visualization.
      * **Scikit-learn:** For the DBSCAN clustering algorithm.
      * **Matplotlib:** For creating plots and charts.

-----

## 5\. Results

The system was tested on the full dataset of 718 frames. The analysis showed that **1069** unique objects were detected and tracked. These included **128** cars, **195** pedestrians, and **79** cyclists.

The tracking algorithm worked well, with **114** objects being tracked for 10 or more frames in a row. The object tracked for the longest time, likely a fixed part of the environment, was followed for **794** frames. This shows the tracker's stability.

<img width="1000" height="600" alt="tracking_durations_histogram" src="https://github.com/user-attachments/assets/4d599b6b-85da-492c-b167-fcfbd0e70d4e" />

-----
