import numpy as np

class ExtendedKalmanFilter:
    def __init__(self, state_dim, R, Q):
        self.state_dim = state_dim
        self.R = R
        self.Q = Q

    @staticmethod
    def state_transition(x, v, dt):
        return x + v * dt

    @staticmethod
    def measurement_model(x, landmarks):
        distances = [np.sqrt((x[0] - l[0])**2 + (x[1] - l[1])**2) for l in landmarks]
        return np.array(distances)

    @staticmethod
    def jacobian(x, h, landmarks):
        jacobian = []
        for l, h_val in zip(landmarks, h):
            row = [(x[0] - l[0]) / h_val, (x[1] - l[1]) / h_val]
            jacobian.append(row)
        return np.array(jacobian)

    def predict(self, x, v, P, dt):
        x_pred = self.state_transition(x, v, dt)
        P_pred = P + self.R
        return x_pred, P_pred

    def update(self, x_pred, P_pred, z, landmarks):
        z_pred = self.measurement_model(x_pred, landmarks)
        H = self.jacobian(x_pred, z_pred, landmarks)

        S = H @ P_pred @ H.T + self.Q
        K = P_pred @ H.T @ np.linalg.inv(S)

        y = z - z_pred
        x_updated = x_pred + K @ y
        P_updated = (np.eye(self.state_dim) - K @ H) @ P_pred

        return x_updated, P_updated
