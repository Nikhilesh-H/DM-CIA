from KalmanFilter import *

# Tracks a single object using a Kalman filter
class KalmanTracker:

    # Initialize tracker with object's position
    def __init__(self, x, y):
        self.kf = KalmanFilter(4, 2)
        self.kf.x = np.array([[x], [y], [0], [0]], dtype=np.float32)
        self.kf.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        self.kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        self.kf.Q = np.eye(4, dtype=np.float32) * 0.03
        self.kf.R = np.eye(2, dtype=np.float32) * 0.1
        self.prediction = self.kf.x.copy()
        self.hits = 0
        self.time_since_update = 0
        self.id = None

    # Predict object's next position
    def predict(self):
        self.prediction = self.kf.predict()
        self.time_since_update += 1
        return self.prediction[:2]

    # Update tracker with new measurement
    def update(self, x: float, y: float):
        measurement = np.array([[x], [y]], dtype=np.float32)
        self.kf.correct(measurement)
        self.hits += 1
        self.time_since_update = 0
        self.prediction = self.kf.x.copy()