import math

class EuclideanDistTracker:
    """
    A simple object tracker that uses Euclidean distance to associate objects
    between frames.
    """
    def __init__(self):
        # Stores the center points of the objects we are currently tracking.
        # Format: {object_id: (center_x, center_y)}
        self.center_points = {}
        # A counter to assign a new, unique ID to each new object.
        self.id_count = 0

    def update(self, objects_rects, dist_thresh=5.0):
        """
        Updates the tracker with new object detections from the current frame.

        Args:
            objects_rects (list): A list of bounding boxes for the newly detected objects.
                                  Each box should be in the format [x, y, w, h].
            dist_thresh (float): The maximum distance (in meters) for a detection to be
                                 matched with an existing object.

        Returns:
            list: A list of the tracked objects, with their assigned IDs.
                  Each object is represented as [x, y, w, h, object_id].
        """
        tracked_objects = []
        
        # Keep track of which new detections have not been matched yet.
        unmatched_detections = list(range(len(objects_rects)))
        
        # If we are already tracking objects, try to match them with the new detections.
        if self.center_points:
            # For each object we are tracking...
            for obj_id, pt in self.center_points.items():
                best_match_idx = -1
                min_dist = dist_thresh
                
                # ...find the closest new detection.
                for i in unmatched_detections:
                    rect = objects_rects[i]
                    x, y, w, h = rect
                    cx = x + w / 2
                    cy = y + h / 2
                    dist = math.hypot(cx - pt[0], cy - pt[1])

                    # If this detection is closer than our current best match, update it.
                    if dist < min_dist:
                        min_dist = dist
                        best_match_idx = i
                
                # If we found a good match...
                if best_match_idx != -1:
                    # ...get the matched detection's info...
                    rect = objects_rects[best_match_idx]
                    x, y, w, h = rect
                    cx = x + w / 2
                    cy = y + h / 2
                    
                    # ...update the tracked object's center point...
                    self.center_points[obj_id] = (cx, cy)
                    # ...add it to our list of tracked objects for this frame...
                    tracked_objects.append([x, y, w, h, obj_id])
                    # ...and remove it from the list of unmatched detections.
                    unmatched_detections.remove(best_match_idx)

        # Any detections left in the unmatched list are considered new objects.
        for i in unmatched_detections:
            rect = objects_rects[i]
            x, y, w, h = rect
            cx = x + w / 2
            cy = y + h / 2
            
            # Assign a new ID to this new object.
            self.id_count += 1
            self.center_points[self.id_count] = (cx, cy)
            tracked_objects.append([x, y, w, h, self.id_count])

        # Clean up the center_points dictionary by removing any IDs that are no longer
        # being tracked (i.e., they disappeared from the frame).
        active_ids = {obj[4] for obj in tracked_objects}
        self.center_points = {obj_id: center for obj_id, center in self.center_points.items() if obj_id in active_ids}

        return tracked_objects