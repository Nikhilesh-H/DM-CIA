import numpy as np

# Implements a basic Kalman filter for state estimation
class KalmanFilter:

    # Initialize the Kalman filter with state and measurement dimensions
    def __init__(self, state_dim, meas_dim):
        self.state_dim = state_dim
        self.meas_dim = meas_dim
        self.x = np.zeros((state_dim, 1), dtype=np.float32)
        self.P = np.eye(state_dim, dtype=np.float32)
        self.F = np.eye(state_dim, dtype=np.float32)
        self.H = np.zeros((meas_dim, state_dim), dtype=np.float32)
        self.R = np.eye(meas_dim, dtype=np.float32)
        self.Q = np.eye(state_dim, dtype=np.float32)

    # Predict the next state based on the motion model
    def predict(self):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x

    # Update state estimate using measurement data
    def correct(self, z):
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        y = z - np.dot(self.H, self.x)
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.state_dim)
        self.P = np.dot((I - np.dot(K, self.H)), self.P)