# Moving Object Detection and Tracking

This project implements moving object detection and tracking in video sequences using frame differencing and Kalman Filters. It identifies motion regions, tracks objects across frames, and visualizes predicted positions in real-time.

## 🔧 Files Included

- `KalmanFilter.py` – Basic Kalman filter for state estimation.
- `KalmanTracker.py` – Uses Kalman filter to track a single object.
- `MovingObjectDetector.py` – Detects motion and manages multiple object trackers.
- `main.py` – Runs the full detection-tracking pipeline on a video.

## 🚀 Features

- Motion detection using frame differencing
- Object tracking with Kalman Filter prediction
- Hungarian algorithm for detection-tracker assignment
- Handles multiple objects and occlusions
- Visualizes object IDs, bounding boxes, motion area, and prediction points

## 📦 Dependencies

- Python 3.x  
- OpenCV  
- NumPy  
- SciPy

Install dependencies:

```bash
pip install opencv-python numpy scipy
python main.py
```
