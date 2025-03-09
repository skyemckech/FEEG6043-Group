import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

"""
This script reads data from a JSON log file that includes:
 - /state topic: The estimated (x, y, yaw)
 - /aruco topic: The measured positions from markers
 - /covariance_pos topic: 2D covariance for the estimated position

It then plots the estimated trajectory in BLUE, the measured positions in GREEN 'x',
and draws RED covariance ellipses (similar to the style in your EKF tutorial,
where the ellipses have a slight fill/transparency in red).

Now, we only plot an ellipse every 5 data points to avoid clutter.
"""

#==============================================================
# ADJUST THIS PATH to match your file location:
#==============================================================
file_path = r"C:\Users\danae\Folder\FEEG6043-Group-2\logs\20250309_155232_log.json"

#==============================================================
# 1. Load the data from the JSON log
#==============================================================
state_list = []       # will store [x, y, yaw]
measured_list = []    # will store [x, y]
covariance_list = []  # will store 2x2 covariance matrices

with open(file_path, 'r') as file:
    for line in file:
        data = json.loads(line)
        topic = data.get("topic_name")

        #--------------------------------------------------------
        # Extract estimated state (x, y, yaw) from /state
        #--------------------------------------------------------
        if topic == "/state":
            vec = data["message"].get("vector", {})
            if all(k in vec for k in ["x", "y", "z"]):
                x_val = vec["x"]
                y_val = vec["y"]
                yaw   = vec["z"]  # z is yaw in your log
                state_list.append([x_val, y_val, yaw])

        #--------------------------------------------------------
        # Extract measured position from /aruco
        #--------------------------------------------------------
        elif topic == "/aruco":
            pose = data["message"].get("pose", {})
            pos  = pose.get("position", {})
            if all(k in pos for k in ["x", "y"]):
                measured_list.append([pos["x"], pos["y"]])

        #--------------------------------------------------------
        # Extract covariance from /covariance_pos
        #--------------------------------------------------------
        elif topic == "/covariance_pos":
            cvec = data["message"].get("vector", {})
            if all(k in cvec for k in ["x", "y", "z"]):
                cov_matrix = np.array([
                    [cvec["x"], cvec["z"]],
                    [cvec["z"], cvec["y"]]
                ])
                covariance_list.append(cov_matrix)

# Convert them to numpy arrays
state_data    = np.array(state_list)
measured_data = np.array(measured_list)

print(f"Loaded {len(state_data)} states, {len(measured_data)} measured points, "
      f"and {len(covariance_list)} covariance matrices.")


#==============================================================
# 2. Define a function to plot the data in the tutorial's style
#==============================================================
def plot_ekf_trajectory(state_arr, measured_arr, cov_list, keyframe=1):
    """
    Plots:
      - The estimated state path in blue with circle markers
      - The measured positions in green 'x'
      - Covariance ellipses in red (semi-transparent fill)

    keyframe: plot an ellipse every 'keyframe' steps (to avoid clutter)
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # 1) Estimated state in BLUE
    if len(state_arr) > 0:
        ax.plot(state_arr[:, 0], state_arr[:, 1], 'bo-', label='Estimated State', markersize=5)

    # 2) Measured position in GREEN 'x'
    if len(measured_arr) > 0:
        ax.plot(measured_arr[:, 0], measured_arr[:, 1], 'gx', label='Measured Position', markersize=6)

    # 3) Covariance ellipses in RED 
    for i, cov in enumerate(cov_list):
        if cov.shape == (2, 2) and i < len(state_arr) and (i % keyframe == 0):
            # compute ellipse from 2D cov
            eigvals, eigvects = np.linalg.eigh(cov)
            angle  = np.degrees(np.arctan2(*eigvects[:, 0][::-1]))
            width  = 2.0 * np.sqrt(abs(eigvals[0]))
            height = 2.0 * np.sqrt(abs(eigvals[1]))

            # red fill, alpha=0.2 
            ell = Ellipse(
                (state_arr[i, 0], state_arr[i, 1]),
                width, height,
                angle=angle,
                edgecolor='red',
                facecolor='red',
                alpha=0.2,
                linestyle='-'
            )
            ax.add_patch(ell)

   # Set labels, title, grid, and aspect ratio
    ax.set_xlabel('Eastings (m)')
    ax.set_ylabel('Northings (m)')
    ax.legend()
    ax.grid(True)
    ax.set_aspect('equal')
    plt.title('EKF Estimated Trajectory with Covariance Ellipses (Tutorial Style)')
    plt.show()

# ==============================================================
# 3. Plot the data

# Basic checks
if len(state_data) == 0:
    print("No state data to plot.")
elif len(measured_data) == 0:
    print("No measured data to plot.")
else:
    # Plot an ellipse every 5th data point
    plot_ekf_trajectory(state_data, measured_data, covariance_list, keyframe=1)
