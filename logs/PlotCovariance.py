import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle

# Path to the JSON file
file_path = "logs/20250309_200252_log.json"

# Initialize data storage
state_data = []
groundtruth_data = []
measured_data = []
covariance_data = []

# Read log file and extract relevant data
with open(file_path, 'r') as file:
    for line in file:
        data = json.loads(line)
        topic = data.get("topic_name")

        # Extract estimated state (position and yaw)
        if topic == "/state":
            state_vector = data["message"]["vector"]
            if all(k in state_vector for k in ["x", "y", "z"]):
                state_data.append([state_vector["x"], state_vector["y"], state_vector["z"]])

        # Extract ground truth position
        elif topic == "/groundtruth":
            pos = data["message"]["position"]
            if "x" in pos and "y" in pos:
                groundtruth_data.append([pos["x"], pos["y"]])

        # Extract measured position from ArUco markers
        elif topic == "/aruco":
            pos = data["message"]["pose"]["position"]
            if "x" in pos and "y" in pos:
                measured_data.append([pos["x"], pos["y"]])

        # Extract covariance matrix for position
        elif topic == "/covariance_pos":
            covariance_vector = data["message"]["vector"]
            if all(k in covariance_vector for k in ["x", "y", "z"]):
                covariance_matrix = np.array([
                    [covariance_vector["x"], covariance_vector["z"]],
                    [covariance_vector["z"], covariance_vector["y"]]
                ])
                covariance_data.append(covariance_matrix)
                print(f"Loaded Covariance Matrix {len(covariance_data)}:\n{covariance_matrix}")

# Convert data to numpy arrays for easier plotting
state_data = np.array(state_data)
groundtruth_data = np.array(groundtruth_data)
measured_data = np.array(measured_data)

print(f"Total covariance matrices loaded: {len(covariance_data)}")

# Check if data arrays are not empty before plotting
if state_data.size > 0 and groundtruth_data.size > 0 and measured_data.size > 0:
    # Plotting
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot the estimated state trajectory
    ax.plot(state_data[:800, 1], state_data[:800, 0], 'bo-', label='Estimated State', markersize=4)

    # Plot the ground truth trajectory
    # ax.plot(groundtruth_data[:, 1], groundtruth_data[:, 0], 'r-', label='Ground Truth', linewidth=2)

    # Plot the measured positions from the ArUco markers
    ax.plot(measured_data[:80, 1], measured_data[:80, 0], 'gx', label='Measured Position', markersize=6)

    # Draw covariance ellipses in the style of the EKF tutorial
    for i, cov_matrix in enumerate(covariance_data):
        if cov_matrix.shape == (2, 2) and i < len(state_data):
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
            width, height = 2 * np.sqrt(np.abs(eigenvalues))
            print(f"Ellipse {i}: Width={width}, Height={height}, Angle={angle}")
            if i == 0:
                ellipse = Ellipse(
                    (state_data[i, 0], state_data[i, 1]),
                    width,
                    height,
                    angle=angle,
                    edgecolor='purple',
                    facecolor='red',
                    linestyle='-',
                    alpha=0.7,
                    label = 'covariance'
                )
            else:
                ellipse = Ellipse(
                (state_data[i, 1], state_data[i, 0]),
                width,
                height,
                angle=angle,
                edgecolor='purple',
                facecolor='red',
                linestyle='-',
                alpha=0.7
            )
            ax.add_patch(ellipse)


    radius = np.sqrt(0.001)  # Radius of the circle
    for (xi, yi) in zip(measured_data[:80, 1], measured_data[:80, 0]):
        circle = Circle((xi, yi), radius, edgecolor='green', facecolor='green', linewidth=2,alpha = 0.3)
        ax.add_patch(circle)
    # Set labels, grid, and aspect ratio
    ax.set_xlabel('Eastings (m)')
    ax.set_ylabel('Northings (m)')
    ax.legend()
    ax.set_aspect('equal')
    plt.grid(True)
    plt.title('EKF Estimated Trajectory with Ground Truth and Covariance Ellipses')
    plt.show()
else:
    print("Insufficient data to plot. Please check the log file for completeness.")
