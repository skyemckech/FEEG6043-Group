import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Path to the JSON file
file_path = "logs/20250309_200252_log.json"

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
        elif topic == "/measured_est_distance":
            pos = data["message"]["vector"]
            if "x" in pos and "y" in pos:
                measured_data.append([pos["x"], pos["y"]])

