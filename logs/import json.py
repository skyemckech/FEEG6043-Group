import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Path to the JSON file
file_path = r"C:\Users\danae\Folder\FEEG6043-Group-2\logs\20250308_183838_log.json"

# Initialize data storage for state, measurements, and covariance matrices
state_data = []  # To store estimated state vectors [x, y, yaw]
measured_data = []  # To store measured positions from ArUco markers
covariance_data = []  # To store covariance matrices for position

# Read log file and extract relevant data
with open(file_path, 'r') as file:
    for line in file:
        data = json.loads(line)
        topic = data.get("topic_name")

        # Extract estimated state (position and yaw)
        if topic == "/state":
            state_vector = data["message"]["vector"]
            if isinstance(state_vector, list) and len(state_vector) >= 3:
                state_data.append([state_vector[0], state_vector[1], state_vector[2]])

        # Extract measured position from ArUco markers
        elif topic == "/aruco":
            pos = data["message"]["pose"]["position"]
            if "x" in pos and "y" in pos:
                measured_data.append([pos["x"], pos["y"]])

        # Extract covariance matrix for position
        elif topic == "/covariance_pos":
            covariance_vector = data["message"]["vector"]
            if isinstance(covariance_vector, list) and len(covariance_vector) >= 3:
                covariance_matrix = np.array([
                    [covariance_vector[0], covariance_vector[2]],
                    [covariance_vector[2], covariance_vector[1]]
                ])
                covariance_data.append(covariance_matrix)

# Convert data to numpy arrays for easier plotting
state_data = np.array(state_data)
measured_data = np.array(measured_data)

# Check if data arrays are not empty before plotting
if state_data.size > 0 and measured_data.size > 0:
    # Plotting
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot the estimated state trajectory if available
    if state_data.shape[1] >= 2:
        ax.plot(state_data[:, 0], state_data[:, 1], 'ro-', label='Estimated State')

    # Plot the measured positions from the ArUco markers
    if measured_data.shape[1] >= 2:
        ax.plot(measured_data[:, 0], measured_data[:, 1], 'gx', label='Measured Position', markersize=10)

    # Draw covariance ellipses
    for i, cov_matrix in enumerate(covariance_data):
        if cov_matrix.shape == (2, 2) and i < len(state_data):
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
            width, height = 2 * np.sqrt(np.abs(eigenvalues))
            ellipse = Ellipse(
                xy=(state_data[i, 0], state_data[i, 1]), 
                width=width, 
                height=height, 
                angle=angle,  # Correctly used as a keyword argument
                edgecolor='purple', 
                facecolor='none', 
                linestyle='--', 
                alpha=0.7
            )
            ax.add_patch(ellipse)

    # Set labels, grid, and aspect ratio for the plot
    ax.set_xlabel('Eastings (m)')
    ax.set_ylabel('Northings (m)')
    ax.legend()
    ax.set_aspect('equal')
    plt.grid(True)
    plt.title('EKF Estimated Trajectory with Position Covariance Ellipses')
    plt.show()
else:
    print("Insufficient data to plot. Please check the log file for completeness.")
