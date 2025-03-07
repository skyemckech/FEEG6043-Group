import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Path to the JSON file
file_path = r"C:/Users/danae/Folder/FEEG6043-Group-2/logs/20250303_190632_log.json"

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

        if topic == "/groundtruth":
            pos = data["message"]["position"]
            groundtruth_data.append([pos["x"], pos["y"]])

        elif topic == "/est_pose":
            pos = data["message"]["pose"]["position"]
            state_data.append([pos["x"], pos["y"]])

        elif topic == "/aruco":
            pos = data["message"]["pose"]["position"]
            measured_data.append([pos["x"], pos["y"]])

        elif topic == "/covariance":
            covariance = data["message"]["covariance"]
            covariance_data.append(covariance)

# Convert to numpy arrays for easier manipulation
state_data = np.array(state_data)
groundtruth_data = np.array(groundtruth_data)
measured_data = np.array(measured_data)
covariance_data = np.array(covariance_data)

# Plotting
fig, ax = plt.subplots(figsize=(6, 6))

# Ground truth
ax.plot(groundtruth_data[:, 0], groundtruth_data[:, 1], 'bo-', label='Ground Truth')

# Estimated State
ax.plot(state_data[:, 0], state_data[:, 1], 'ro-', label='Estimated State')

# Measured Position
ax.plot(measured_data[:, 0], measured_data[:, 1], 'gx', label='Measured Position', markersize=10)

# Covariance as ellipses
for i, cov in enumerate(covariance_data):
    if i % 10 == 0:  # Reduce the number of ellipses for clarity
        width, height = 2 * np.sqrt(cov[0][0]), 2 * np.sqrt(cov[1][1])
        ellipse = Ellipse((state_data[i, 0], state_data[i, 1]), width, height,
                          edgecolor='pink', facecolor='none', alpha=0.6)
        ax.add_patch(ellipse)

# Labels and Legend
ax.set_xlabel('Eastings (m)')
ax.set_ylabel('Northings (m)')
ax.legend()
ax.set_aspect('equal')
plt.grid(True)
plt.show()

