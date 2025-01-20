import numpy as np
import matplotlib.pyplot as plt
from ekf_nav.ekf import ExtendedKalmanFilter
from ekf_nav.utils import generate_velocity_sequence, add_noise

def plot_positions(true_positions, estimated_positions, P, landmarks):
    sigma_bounds = 3 * np.sqrt(np.diag(P))

    plt.figure(figsize=(10, 10))
    plt.plot(true_positions[:, 0], true_positions[:, 1], label='True Position', color='blue', linewidth=2)
    plt.plot(estimated_positions[:, 0], estimated_positions[:, 1], label='Estimated Position', color='red', linestyle='--', linewidth=2)

    for i in range(len(estimated_positions)):
        plt.errorbar(estimated_positions[i, 0], estimated_positions[i, 1],
                     xerr=sigma_bounds[0], yerr=sigma_bounds[1], fmt='o', color='red', alpha=0.5)

    for i, landmark in enumerate(landmarks):
        plt.scatter(landmark[0], landmark[1], label=f'Landmark {i + 1}', color='green', marker='x', s=100)

    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('True and Estimated Robot Positions with 3σ Confidence Bounds')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    dt = 0.5
    R = np.array([[0.1, 0.0], [0.0, 0.1]])  # State transition covariance
    Q = np.array([[0.5, 0.0], [0.0, 0.5]])  # Measurement noise covariance
    P = np.eye(2)  # Initial covariance matrix

    x = np.array([0.0, 0.0])  # Initial position
    landmarks = [np.array([5, 5]), np.array([-5, 5])]
    vel_seq = [(1, 0), (0, -1), (-1, 0), (0, 1)]

    ekf = ExtendedKalmanFilter(state_dim=2, R=R, Q=Q)

    true_positions = []
    estimated_positions = []

    for t in np.arange(0, 40.5, dt):
        # Generate velocity
        v = generate_velocity_sequence(t, vel_seq)

        # True motion with noise
        x = x + v * dt
        x = add_noise(x, R[0, 0])

        # EKF prediction
        if t == 0:
            x_pred, P_pred = ekf.predict(x, v, P, dt)
        else:
            x_pred, P_pred = ekf.predict(x_pred, v, P_pred, dt)

        # Measurement generation
        z = ekf.measurement_model(x, landmarks)
        z = add_noise(z, Q[0, 0])

        # EKF update
        x_pred, P_pred = ekf.update(x_pred, P_pred, z, landmarks)

        # Store positions
        true_positions.append(x)
        estimated_positions.append(x_pred)

    # Plot results
    plot_positions(np.array(true_positions), np.array(estimated_positions), P_pred, landmarks)

if __name__ == "__main__":
    main()
