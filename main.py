from KalmanTracker import *
from MovingObjectDetector import *

def main():

    detector = MovingObjectDetector(
        threshold=5,
        history_size=3,
        min_area=0.01,
        max_distance=0.2,
        inactive_threshold=800,
        tracking_duration=5.0,
        max_missed_frames=10
    )
    
    cap = cv2.VideoCapture("videos/video-6.mp4")

    while True:
        ret, frame = cap.read()
        if not ret: break
        labeled_frame, binary_mask = detector.process_frame(frame)
        cv2.imshow('Labeled Objects', labeled_frame)
        cv2.imshow('Motion Mask', binary_mask)
        if cv2.waitKey(20) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()