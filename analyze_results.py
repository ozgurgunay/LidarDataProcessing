import json
import os
from glob import glob
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = "output"
json_files = sorted(glob(os.path.join(OUTPUT_DIR, "*.json")))

if not json_files:
    print(f"Warning: Could not find any JSON files to analyze in the '{OUTPUT_DIR}' directory.")
    exit()

all_detections = []
# Read all data from the JSON files
for file_path in json_files:
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            all_detections.extend(data)
        except json.JSONDecodeError:
            print(f"WARNING: The file {file_path} is corrupted or empty. Skipping.")

print(f"--- Analysis of {len(json_files)} frames completed ---")

# (The analysis and print sections from the original code are here)

# --- GENERATE PLOTS ---

# 1. bar chart for unique object counts
object_id_to_classes = {}
for det in all_detections:
    obj_id = det.get('object_id', -1)
    if obj_id != -1:
        if obj_id not in object_id_to_classes:
            object_id_to_classes[obj_id] = []
        object_id_to_classes[obj_id].append(det['class'])

unique_objects_final_class = {}
for obj_id, class_list in object_id_to_classes.items():
    most_common_class = Counter(class_list).most_common(1)[0][0]
    unique_objects_final_class[obj_id] = most_common_class

if unique_objects_final_class:
    final_unique_counts = Counter(unique_objects_final_class.values())
    
    # Prepare data for the plot
    labels = list(final_unique_counts.keys())
    counts = list(final_unique_counts.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, counts, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    plt.title('Count of Unique Objects Detected Across All Frames')
    plt.ylabel('Number of Unique Objects')
    plt.xlabel('Object Class')
    for bar in bars:
        yval = bar.get_height()
        # Add the exact count on top of each bar
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, int(yval), va='bottom') 
    
    plt.savefig('unique_object_counts.png')
    print("\nGenerated plot: 'unique_object_counts.png'")

# 2. Histogram of tracking durations
valid_ids = [det.get('object_id', -1) for det in all_detections if det.get('object_id', -1) != -1]
if valid_ids:
    id_counts = Counter(valid_ids)
    durations = list(id_counts.values())
    
    plt.figure(figsize=(10, 6))
    plt.hist(durations, bins=30, color='skyblue', edgecolor='black')
    plt.title('Distribution of Object Tracking Durations')
    plt.xlabel('Number of Frames an Object was Tracked (Tracking Duration)')
    plt.ylabel('Number of Objects')
    # Use a log scale for the y-axis to better visualize the distribution,
    # as there are many short-lived tracks.
    plt.yscale('log') 
    
    plt.savefig('tracking_durations_histogram.png')
    print("Generated plot: 'tracking_durations_histogram.png'")