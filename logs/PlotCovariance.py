import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Path to the JSON file
file_path = r"C:/Users/danae/Folder/FEEG6043-Group-2/logs/20250307_192906_log.json"

# Initialize data storage
state_data = []
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

# Convert data to numpy arrays for easier plotting
state_data = np.array(state_data)
measured_data = np.array(measured_data)

# Check if data arrays are not empty before plotting
if state_data.size > 0 and measured_data.size > 0:
    # Plotting
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot the estimated state trajectory
    ax.plot(state_data[:, 0], state_data[:, 1], 'ro-', label='Estimated State')

    # Plot the measured positions from the ArUco markers
    ax.plot(measured_data[:, 0], measured_data[:, 1], 'gx', label='Measured Position', markersize=10)

    # Draw covariance ellipses
    for i, cov_matrix in enumerate(covariance_data):
        if cov_matrix.shape == (2, 2) and i < len(state_data):
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
            width, height = 2 * np.sqrt(np.abs(eigenvalues))
            ellipse = Ellipse((state_data[i, 0], state_data[i, 1]), width, height, angle,
                              edgecolor='purple', facecolor='none', linestyle='--', alpha=0.7)
            ax.add_patch(ellipse)

    # Set labels, grid, and aspect ratio
    ax.set_xlabel('Eastings (m)')
    ax.set_ylabel('Northings (m)')
    ax.legend()
    ax.set_aspect('equal')
    plt.grid(True)
    plt.title('EKF Estimated Trajectory with Position Covariance Ellipses')
    plt.show()
else:
    print("Insufficient data to plot. Please check the log file for completeness.")
