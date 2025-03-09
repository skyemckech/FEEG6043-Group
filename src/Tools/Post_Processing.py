import json
import numpy as np
import math
import matplotlib.pyplot as plt

def quaternion_to_heading(heading_w, heading_x, heading_y, heading_z):
    """
    Converts a quaternion (w, x, y, z) to heading (yaw) in radians.

    :param heading_w: scalar part of the quaternion from log data
    :param heading_x: x-component of the quaternion from log data
    :param heading_y: y-component of the quaternion from log data
    :param heading_z: z-component of the quaternion from log data
    :return: heading (yaw) in radians
    """
    heading = math.atan2(2 * (heading_w * heading_z + heading_x * heading_y), 1 - 2 * (heading_y**2 + heading_z**2))
    return heading

class ImportLog:
    def __init__(self, filepath):
        # Expecting a file path to initialize
        self.filepath = filepath
        self.data = {}
        self._parse_log()

    def _parse_log(self):
        """Reads the JSON log file and organizes data by topic name."""
        with open(self.filepath, 'r') as file:
            for line in file:
                try:
                    entry = json.loads(line.strip())
                    topic = entry.get("topic_name", "unknown")
                    if topic not in self.data:
                        self.data[topic] = []
                    self.data[topic].append(entry)
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line: {line.strip()}")

    def get_data_by_topic(self, topic_name):
        """Returns all log entries for a specific topic."""
        return self.data.get(topic_name, [])

    def extract_data(self, topic_name, nested_keys):
        """
        Extracts specified data from a given topic.
        
        :param topic_name: The name of the topic to extract from (e.g., "/est_pose").
        :param nested_keys: A list of keys specifying the path to the desired data.
                            Example: ["message", "pose", "position"]
        :return: A list of extracted values.
        """
        extracted_data = []
        for entry in self.get_data_by_topic(topic_name):
            data = entry
            for key in nested_keys:
                data = data.get(key, {})  # Keep traversing down the dictionary
            if isinstance(data, dict):  # If it's a dict, extract values as tuple
                extracted_data.append(tuple(data.get(k) for k in data))
            else:  # If it's a single value, just append it
                extracted_data.append(data)
    
        return extracted_data

# Standalone function to extract variables
def extract_variables(filepath):
    # Create an ImportLog instance by passing the file path to ImportLog
    variables = ImportLog(filepath)  # Passing the file path correctly

    # Extract data using the extract_data method of the ImportLog instance
    northings_ground = variables.extract_data("/groundtruth", ["message","position", "x"])
    eastings_ground = variables.extract_data("/groundtruth", ["message", "position", "y"])
    heading_x = variables.extract_data("/groundtruth", ["message", "orientation", "x"])
    heading_y = variables.extract_data("/groundtruth", ["message", "orientation", "y"])
    heading_z = variables.extract_data("/groundtruth", ["message", "orientation", "z"])
    heading_w = variables.extract_data("/groundtruth", ["message", "orientation", "w"])
    wheel_right = variables.extract_data("/true_wheel_speeds", ["message", "vector", "x"])
    wheel_left = variables.extract_data("/true_wheel_speeds", ["message", "vector", "y"])
    timestamps = variables.extract_data("/groundtruth", ["timestamp"])

    headings = [quaternion_to_heading(w, x, y, z) for w, x, y, z in zip(heading_w, heading_x, heading_y, heading_z)]
    # Convert timestamps from strings to floats
    timestamps = [float(timestamp) for timestamp in timestamps]
    
    # Compute angular velocity by differentiating heading over time
    angular_velocity = []
    for i in range(1, len(headings)):
        delta_t = timestamps[i] - timestamps[i-1]  # Time difference between consecutive timestamps
        if delta_t != 0:
            angular_velocity.append((headings[i] - headings[i-1]) / delta_t)  # Angular velocity in rad/s
        else:
            angular_velocity.append(0)  # If delta_t is zero, assume no angular velocity
    
    return {
        "timestamps": timestamps,
        "northings_ground": northings_ground,
        "eastings_ground": eastings_ground,
        "headings": headings,
        "wheel_right": wheel_right,
        "wheel_left": wheel_left,
        "angular_velocity": angular_velocity
    }


# def construct_R_matrix(extracted_data):
#     # Variance of the states
#     variances = compute_variance(extracted_data)  # This should give you the variances of the states
    
#     # Variance values (diagonal elements of R)
#     northings_variance = variances["northings_ground"]
#     eastings_variance = variances["eastings_ground"]
#     forward_velocity_variance = variances["forward_velocity"]
#     angular_velocity_variance = variances["angular_velocity"]
#     heading_variance = variances["heading"]
    
#     # Find the minimum length across all the data arrays
#     min_length = min(len(extracted_data["northings_ground"]),
#                      len(extracted_data["eastings_ground"]),
#                      len(extracted_data["forward_velocity"]),
#                      len(extracted_data["angular_velocity"]),
#                      len(extracted_data["headings"]))

#     # Trim each array to the same minimum length
#     extracted_data["northings_ground"] = extracted_data["northings_ground"][:min_length]
#     extracted_data["eastings_ground"] = extracted_data["eastings_ground"][:min_length]
#     extracted_data["forward_velocity"] = extracted_data["forward_velocity"][:min_length]
#     extracted_data["angular_velocity"] = extracted_data["angular_velocity"][:min_length]
#     extracted_data["headings"] = extracted_data["headings"][:min_length]

#     # Now compute the covariance matrix
#     cov_northings_eastings = np.cov(extracted_data["northings_ground"], extracted_data["eastings_ground"])[0, 1]
#     cov_northings_velocity = np.cov(extracted_data["northings_ground"], extracted_data["forward_velocity"])[0, 1]
#     cov_eastings_velocity = np.cov(extracted_data["eastings_ground"], extracted_data["forward_velocity"])[0, 1]
#     cov_northings_angular_velocity = np.cov(extracted_data["northings_ground"], extracted_data["angular_velocity"])[0, 1]
#     cov_eastings_angular_velocity = np.cov(extracted_data["eastings_ground"], extracted_data["angular_velocity"])[0, 1]
#     cov_velocity_angular_velocity = np.cov(extracted_data["forward_velocity"], extracted_data["angular_velocity"])[0, 1]
#     cov_northings_heading = np.cov(extracted_data["northings_ground"], extracted_data["headings"])[0, 1]
#     cov_eastings_heading = np.cov(extracted_data["eastings_ground"], extracted_data["headings"])[0, 1]
#     cov_velocity_heading = np.cov(extracted_data["forward_velocity"], extracted_data["headings"])[0, 1]
#     cov_angular_velocity_heading = np.cov(extracted_data["angular_velocity"], extracted_data["headings"])[0, 1]

#     # Construct the R matrix
#     R_matrix = np.array([
#         [northings_variance, cov_northings_eastings, cov_northings_velocity, cov_northings_angular_velocity, cov_northings_heading],
#         [cov_northings_eastings, eastings_variance, cov_eastings_velocity, cov_eastings_angular_velocity, cov_eastings_heading],
#         [cov_northings_velocity, cov_eastings_velocity, forward_velocity_variance, cov_velocity_angular_velocity, cov_velocity_heading],
#         [cov_northings_angular_velocity, cov_eastings_angular_velocity, cov_velocity_angular_velocity, angular_velocity_variance, cov_angular_velocity_heading],
#         [cov_northings_heading, cov_eastings_heading, cov_velocity_heading, cov_angular_velocity_heading, heading_variance]
#     ])

    
#     return R_matrix

#%% preparing test files for analysis
# step 1: load the test files
filepaths = ["logs/20250303_190632_log.json", "logs/20250303_190506_log.json",
             "logs/20250303_190426_log.json", "logs/20250303_190345_log.json", "logs/20250303_190300_log.json"]
    
test_data = [extract_variables(fp) for fp in filepaths]

normalized_timestamps = []
for test in test_data:
    timestamps = test["timestamps"]  # Extract timestamps for this test
    start_time = timestamps[0]  # Get the first timestamp of the test
    normalized_timestamps.append([t - start_time for t in timestamps])  # Normalize timestamps

for idx, test in enumerate(test_data):
    timestamps = test["timestamps"]  # Extract timestamps for this test
    time_diffs = np.diff(timestamps)  # Compute time intervals

    # Check if the differences are approximately constant
    mean_diff = np.mean(time_diffs)
    std_dev = np.std(time_diffs)

    print(f"Test {idx}: Mean Interval = {mean_diff:.6f}, Std Dev = {std_dev:.6f}")

# step 2: align the data based on relative timestamps
min_length = min(len(data["timestamps"]) for data in test_data)  # Ensure all tests have the same length

# add checks for other variables, if needed (e.g., angular_velocity, forward_velocity, etc.)
for key in ["northings_ground", "eastings_ground", "headings", "wheel_right", "wheel_left","angular_velocity"]:
    min_length = min(min_length, *[len(data[key]) for data in test_data])

aligned_data = {  # Dictionary to hold aligned states
    "northings": [],
    "eastings": [],
    "headings": [],
    "wheel_right": [],
    "wheel_left": [],
    "angular_velocity": []
}

# check all data is of same length to not throw errors
for idx, data in enumerate(test_data):
    print(f"Test {idx}:")
    print(f"  Timestamps: {len(data['timestamps'])}")
    print(f"  Northings: {len(data['northings_ground'])}")
    print(f"  Eastings: {len(data['eastings_ground'])}")
    print(f"  Headings: {len(data['headings'])}")
    print(f"  wheel right: {len(data['wheel_right'])}")
    print(f"  Wheel left: {len(data['wheel_left'])}")
    print(f"  Angular Velocity: {len(data['angular_velocity'])}")


# stack trials together for each state at each timestamp
for i in range(min_length):
    aligned_data["northings"].append([test_data[j]["northings_ground"][i] for j in range(4)])
    aligned_data["eastings"].append([test_data[j]["eastings_ground"][i] for j in range(4)])
    aligned_data["headings"].append([test_data[j]["headings"][i] for j in range(4)])
    aligned_data["wheel_right"].append([test_data[j]["wheel_right"][i] for j in range(4)])
    aligned_data["wheel_left"].append([test_data[j]["wheel_left"][i] for j in range(4)])
    aligned_data["angular_velocity"].append([test_data[j]["angular_velocity"][i] for j in range(4)])

print("Data successfully aligned. Ready for variance computation!")

#%% analysis and plotting of var
# initialize a dictionary to store variances for each variable
variances = {
    "northings": [],
    "eastings": [],
    "headings": [],
    "wheel_right": [],
    "wheel_left": [],
    "angular_velocity": []
}

# iterate over the aligned data and compute variance for each variable at each timestamp
for i in range(min_length):
    # for each timestamp, calculate the variance of each state (e.g., northings, eastings, etc.)
    northings_values = [aligned_data["northings"][i][j] for j in range(4)]
    eastings_values = [aligned_data["eastings"][i][j] for j in range(4)]
    headings_values = [aligned_data["headings"][i][j] for j in range(4)]
    wheel_right_values = [aligned_data["wheel_right"][i][j] for j in range(4)]
    wheel_left_values = [aligned_data["wheel_left"][i][j] for j in range(4)]
    angular_velocity_values = [aligned_data["angular_velocity"][i][j] for j in range(4)]
    
    # compute variance for each variable
    variances["northings"].append(np.var(northings_values, ddof=1))
    variances["eastings"].append(np.var(eastings_values, ddof=1))
    variances["headings"].append(np.var(headings_values, ddof=1))
    variances["wheel_right"].append(np.var(wheel_right_values, ddof=1))
    variances["wheel_left"].append(np.var(wheel_left_values, ddof=1))
    variances["angular_velocity"].append(np.var(angular_velocity_values, ddof=1))

time = np.arange(min_length)
# Plot variances for each state
plt.figure(figsize=(10, 6))

plt.subplot(2, 3, 1)
plt.plot(time, variances["northings"], label="Northings")
plt.title("Variance of Northings")
plt.xlabel("Time (s)")
plt.ylabel("Variance")

plt.subplot(2, 3, 2)
plt.plot(time, variances["eastings"], label="Eastings", color='orange')
plt.title("Variance of Eastings")
plt.xlabel("Time (s)")
plt.ylabel("Variance")

plt.subplot(2, 3, 3)
plt.plot(time, variances["headings"], label="Headings", color='green')
plt.title("Variance of Headings")
plt.xlabel("Time (s)")
plt.ylabel("Variance")

plt.subplot(2, 3, 4)
plt.plot(time, variances["wheel_right"], label="Measured right wheel rate", color='red')
plt.title("Variance of Measured right wheel rate")
plt.xlabel("Time (s)")
plt.ylabel("Variance")

plt.subplot(2, 3, 5)
plt.plot(time, variances["wheel_left"], label="Measured left wheel rate", color='black')
plt.title("Variance of Measured left wheel rate")
plt.xlabel("Time (s)")
plt.ylabel("Variance")

plt.subplot(2, 3, 6)
plt.plot(time, variances["angular_velocity"], label="Angular Velocity", color='purple')
plt.title("Variance of Angular Velocity")
plt.xlabel("Time (s)")
plt.ylabel("Variance")

plt.tight_layout()
plt.show()

# #%% analysis and plotting of cov
# covariances = {
#     "northings_eastings": [],
#     "northings_headings": [],
#     "northings_forward_velocity": [],
#     "northings_angular_velocity": [],
#     "eastings_headings": [],
#     "eastings_forward_velocity": [],
#     "eastings_angular_velocity": [],
#     "headings_forward_velocity": [],
#     "headings_angular_velocity": [],
#     "forward_velocity_angular_velocity": []
# }

# # Iterate over the aligned data and compute covariance for each pair of variables at each timestamp
# for i in range(min_length):
#     # For each timestamp, extract values for each variable
#     northings_values = [aligned_data["northings"][i][j] for j in range(4)]
#     eastings_values = [aligned_data["eastings"][i][j] for j in range(4)]
#     headings_values = [aligned_data["headings"][i][j] for j in range(4)]
#     forward_velocity_values = [aligned_data["forward_velocity"][i][j] for j in range(4)]
#     angular_velocity_values = [aligned_data["angular_velocity"][i][j] for j in range(4)]

#     # Compute covariance for each pair of variables at each timestamp
#     covariances["northings_eastings"].append(np.cov(northings_values, eastings_values)[0, 1])
#     covariances["northings_headings"].append(np.cov(northings_values, headings_values)[0, 1])
#     covariances["northings_forward_velocity"].append(np.cov(northings_values, forward_velocity_values)[0, 1])
#     covariances["northings_angular_velocity"].append(np.cov(northings_values, angular_velocity_values)[0, 1])
    
#     covariances["eastings_headings"].append(np.cov(eastings_values, headings_values)[0, 1])
#     covariances["eastings_forward_velocity"].append(np.cov(eastings_values, forward_velocity_values)[0, 1])
#     covariances["eastings_angular_velocity"].append(np.cov(eastings_values, angular_velocity_values)[0, 1])

#     covariances["headings_forward_velocity"].append(np.cov(headings_values, forward_velocity_values)[0, 1])
#     covariances["headings_angular_velocity"].append(np.cov(headings_values, angular_velocity_values)[0, 1])

#     covariances["forward_velocity_angular_velocity"].append(np.cov(forward_velocity_values, angular_velocity_values)[0, 1])

# # Plot covariance between pairs of variables
# plt.figure(figsize=(12, 8))

# # Define the pairs of variables to plot
# covariance_pairs = [
#     ("northings_eastings", "Northings vs Eastings"),
#     ("northings_headings", "Northings vs Heading"),
#     ("northings_forward_velocity", "Northings vs Forward Velocity"),
#     ("northings_angular_velocity", "Northings vs Angular Velocity"),
#     ("eastings_headings", "Eastings vs Heading"),
#     ("eastings_forward_velocity", "Eastings vs Forward Velocity"),
#     ("eastings_angular_velocity", "Eastings vs Angular Velocity"),
#     ("headings_forward_velocity", "Heading vs Forward Velocity"),
#     ("headings_angular_velocity", "Heading vs Angular Velocity"),
#     ("forward_velocity_angular_velocity", "Forward Velocity vs Angular Velocity")
# ]

# for key, label in covariance_pairs:
#     plt.figure()  # Create a new figure for each covariance pair
#     plt.plot(covariances[key], label=label)
#     plt.title(label)
#     plt.xlabel("Timestamp")
#     plt.ylabel("Covariance")
#     plt.legend()
#     plt.grid(True)
#     plt.show()  # Show the plot for this covariance pair






























