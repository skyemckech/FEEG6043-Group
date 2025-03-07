import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Path to the JSON file
file_path = r"C:/Users/danae/Folder/FEEG6043-Group-2/logs/20250303_190632_log.json"

# Initialize data storage
state_data = []
measured_data = []
covariance_data = []

# Read log file and extract relevant data
with open(file_path, 'r') as file:
    for line in file:
        data = json.loads(line)
        topic = data.get("topic_name")

        if topic == "/est_pose":
            pos = data["message"]["pose"]["position"]
            state_data.append([pos["x"], pos["y"]])
            if "covariance" in data["message"]:
                covariance_matrix = np.array(data["message"]["covariance"]).reshape(5, 5)
                covariance_data.append(covariance_matrix[:2, :2])

        elif topic == "/covariance":
            covariance_matrix = np.array(data["message"]["covariance"]).reshape(5, 5)
            covariance_data.append(covariance_matrix[:2, :2])

        elif topic == "/aruco":
            pos = data["message"]["pose"]["position"]
            measured_data.append([pos["x"], pos["y"]])

# Convert to numpy arrays
state_data = np.array(state_data)
measured_data = np.array(measured_data)

# Plotting
fig, ax = plt.subplots(figsize=(6, 6))

# Estimated State
ax.plot(state_data[:, 0], state_data[:, 1], 'ro-', label='Estimated State')

# Measured Position
ax.plot(measured_data[:, 0], measured_data[:, 1], 'gx', label='Measured Position', markersize=10)

# Covariance as ellipse (for all logged covariances)
for i, cov_matrix in enumerate(covariance_data):
    if cov_matrix.shape == (2, 2):
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
        width, height = 2 * np.sqrt(np.abs(eigenvalues))
        ellipse = Ellipse((state_data[i, 0], state_data[i, 1]), width, height, angle,
                          edgecolor='purple', facecolor='none', linestyle='--', alpha=0.7)
        ax.add_patch(ellipse)

# Labels and Legend
ax.set_xlabel('Eastings (m)')
ax.set_ylabel('Northings (m)')
ax.legend()
ax.set_aspect('equal')
plt.grid(True)
plt.show()
