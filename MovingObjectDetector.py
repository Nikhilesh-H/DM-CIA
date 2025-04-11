import cv2
from KalmanTracker import *
import time
from scipy.optimize import linear_sum_assignment

# Detects and tracks moving objects in video frames
class MovingObjectDetector:

    # Initialize detector with tracking parameters
    def __init__(self, threshold = 30, history_size = 3, min_area = 0.01, max_distance = 0.2, inactive_threshold = 30, tracking_duration = 5.0, max_missed_frames = 10):
        self.threshold = threshold
        self.history_size = history_size
        self.frame_history = []
        self.min_area = min_area
        self.max_distance = max_distance
        self.inactive_threshold = inactive_threshold
        self.tracking_duration = tracking_duration
        self.max_missed_frames = max_missed_frames
        self.objects = {}
        self.trackers = {}
        self.next_object_id = 1
        self.last_update_time = time.time()
    
    # Preprccessing a frame by converting to grayscale and blur it
    def preprocess_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return blurred
    
    # Calculate absolute difference between two frames    
    def frame_difference(self, frame1, frame2):
        return np.abs(frame1.astype(np.int16) - frame2.astype(np.int16))
    
    # Add up all frame differences
    def sum_differences(self, differences):
        return np.sum(differences, axis=0)
    
    # Create binary mask where motion exceeds threshold
    def threshold_difference(self, diff_sum):
        binary = np.zeros_like(diff_sum, dtype=np.uint8)
        binary[diff_sum > self.threshold] = 255
        return binary
    
    # Clean up the binary mask using morphological operations
    def morphological_operations(self, binary):
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        return cleaned
    
    # Find object contours in binary mask
    def find_contours(self, binary):
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    
    # Calculate center point of a contour
    def calculate_centroid(self, contour):
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            return (cx, cy)
        return (0, 0)
    
    # Calculate normalized distance between two points
    def distance_between_points(self, p1, p2, frame_size):
        dx = (p1[0] - p2[0]) / frame_size[0]
        dy = (p1[1] - p2[1]) / frame_size[1]
        return np.sqrt(dx*dx + dy*dy)
    
    # Match detections with existing trackers and update tracking info
    def update_tracking(self, contours, frame_size):

        current_time = time.time()
        detections = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area / (frame_size[0] * frame_size[1]) < self.min_area:
                continue
            centroid = self.calculate_centroid(contour)
            detections.append((centroid, contour))

        for tracker_id, tracker in self.trackers.items():
            tracker.predict()

        row_ind = []
        col_ind = []

        if len(detections) > 0 and len(self.trackers) > 0:

            cost_matrix = np.zeros((len(detections), len(self.trackers)))
            
            for i, (detection, _) in enumerate(detections):
                for j, (tracker_id, tracker) in enumerate(self.trackers.items()):
                    cost_matrix[i, j] = self.distance_between_points(
                        detection, tracker.prediction[:2].flatten(), frame_size)
                    
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            for i, j in zip(row_ind, col_ind):
                if cost_matrix[i, j] < self.max_distance:
                    tracker_id = list(self.trackers.keys())[j]
                    detection, contour = detections[i]
                    self.trackers[tracker_id].update(*detection)
                    if tracker_id in self.objects:
                        self.objects[tracker_id].update({
                            'found': True,
                            'centroid': detection,
                            'contour': contour,
                            'last_seen': current_time,
                            'inactive_frames': 0
                        })

        for i, (detection, contour) in enumerate(detections):
            if i not in row_ind:
                tracker_id = self.next_object_id
                self.trackers[tracker_id] = KalmanTracker(*detection)
                self.objects[tracker_id] = {
                    'found': True,
                    'centroid': detection,
                    'contour': contour,
                    'last_seen': current_time,
                    'inactive_frames': 0
                }
                self.next_object_id += 1

        for tracker_id, tracker in self.trackers.items():
            if tracker_id not in [list(self.trackers.keys())[j] for j in col_ind]:
                if tracker_id in self.objects:
                    self.objects[tracker_id]['found'] = False
                    self.objects[tracker_id]['inactive_frames'] += 1

        stale_trackers = []

        for tracker_id, tracker in self.trackers.items():
            if tracker.time_since_update > self.max_missed_frames:
                stale_trackers.append(tracker_id)

        for tracker_id in stale_trackers:
            del self.trackers[tracker_id]
            if tracker_id in self.objects:
                del self.objects[tracker_id]

    # Draw bounding boxes and labels on detected objects    
    def label_objects(self, frame, binary):

        labeled_frame = frame.copy()

        for obj_id, obj in self.objects.items():

            x, y, w, h = cv2.boundingRect(obj['contour'])
            if obj['found']: color = (0, 255, 0)
            else: color = (0, 165, 255)

            cv2.rectangle(labeled_frame, (x, y), (x + w, y + h), color, 2)
            label = f"Object {obj_id}"

            cv2.putText(labeled_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            motion_area = np.sum(binary == 255)/(binary.shape[0] * binary.shape[1])
            
            cv2.putText(labeled_frame, f"Area: {motion_area:.2f}", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            status = "Active" if obj['found'] else "Tracking"
            
            cv2.putText(labeled_frame, status, (x, y + h + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            if obj_id in self.trackers and self.trackers[obj_id].prediction is not None:
                try:
                    pred_x, pred_y = self.trackers[obj_id].prediction[:2].flatten()
                    cv2.circle(labeled_frame, (int(pred_x), int(pred_y)), 5, (255, 0, 0), -1)
                except: pass

        return labeled_frame
    
    # Main processing function that handles each video frame    
    def process_frame(self, frame):
        processed = self.preprocess_frame(frame)
        self.frame_history.append(processed)
        if len(self.frame_history) > self.history_size: self.frame_history.pop(0)
        if len(self.frame_history) < 2: return frame, np.zeros_like(processed)
        differences = []
        for i in range(len(self.frame_history) - 1):
            diff = self.frame_difference(self.frame_history[i], self.frame_history[i + 1])
            differences.append(diff)
        diff_sum = self.sum_differences(differences)
        binary = self.threshold_difference(diff_sum)
        binary = self.morphological_operations(binary)
        contours = self.find_contours(binary)
        self.update_tracking(contours, (frame.shape[1], frame.shape[0]))
        labeled_frame = self.label_objects(frame, binary)
        return labeled_frame, binary