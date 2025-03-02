from Libraries.math_feeg6043 import Matrix, Identity
import json, math
import numpy as np

def quaternion_to_heading(heading_w, heading_x, heading_y, heading_z):
    """
    Converts a quaternion (w, x, y, z) to heading (yaw) in radians.

    :param heading_w: scalar part of the quaternion
    :param heading_x: x-component of the quaternion
    :param heading_y: y-component of the quaternion
    :param heading_z: z-component of the quaternion
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
    northings_ground = variables.extract_data("/groundtruth", ["message", "position", "x"])
    eastings_ground = variables.extract_data("/groundtruth", ["message", "position", "y"])
    heading_x = variables.extract_data("/groundtruth", ["message", "orientation", "x"])
    heading_y = variables.extract_data("/groundtruth", ["message", "orientation", "y"])
    heading_z = variables.extract_data("/groundtruth", ["message", "orientation", "z"])
    heading_w = variables.extract_data("/groundtruth", ["message", "orientation", "w"])
    forward_velocity = variables.extract_data("/true_wheel_speeds", ["message", "vector", "x"])
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
        "northings_ground": northings_ground,
        "eastings_ground": eastings_ground,
        "heading_x": heading_x,
        "heading_y": heading_y,
        "heading_z": heading_z,
        "heading_w": heading_w,
        "forward_velocity": forward_velocity,
        "headings": headings,
        "angular_velocity": angular_velocity
    }

# Standalone function to compute the mean of extracted variables
def compute_means(extracted_data):
    headings = extracted_data.get("headings", [])
    stats = {
        "northings_ground_mean": np.mean(extracted_data["northings_ground"]) if extracted_data["northings_ground"] else None,
        "eastings_ground_mean": np.mean(extracted_data["eastings_ground"]) if extracted_data["eastings_ground"] else None,
        "heading": np.mean(headings),
        "forward_velocity_mean": np.mean(extracted_data["forward_velocity"]) if extracted_data["forward_velocity"] else None,
        "angular_velocity_mean": np.mean(extracted_data["angular_velocity"]) if extracted_data["angular_velocity"] else None,
    }
    
    return stats

def compute_variance(extracted_data):
    variances = {
        "northings_ground": np.var(extracted_data["northings_ground"]) if extracted_data["northings_ground"] else None,
        "eastings_ground": np.var(extracted_data["eastings_ground"]) if extracted_data["eastings_ground"] else None,
        "heading": np.var(extracted_data["headings"]) if extracted_data["headings"] else None,
        "forward_velocity": np.var(extracted_data["forward_velocity"]) if extracted_data["forward_velocity"] else None,
        "angular_velocity": np.var(extracted_data["angular_velocity"]) if extracted_data["angular_velocity"] else None
        
    }
    return variances

def compute_covariance(extracted_data):
    # Only compute covariance for the 5 states
    data = [
        extracted_data["northings_ground"],
        extracted_data["eastings_ground"],
        extracted_data["heading"],
        extracted_data["forward_velocity"],
        extracted_data["angular_velocity"]
    ]
    
    # Compute covariance matrix for the 5 states
    covariance_matrix = np.cov(data)
    
    return covariance_matrix

def construct_R_matrix(extracted_data):
    # Variance of the states
    variances = compute_variance(extracted_data)  # This should give you the variances of the states
    
    # Variance values (diagonal elements of R)
    northings_variance = variances["northings_ground"]
    eastings_variance = variances["eastings_ground"]
    forward_velocity_variance = variances["forward_velocity"]
    angular_velocity_variance = variances["angular_velocity"]
    heading_variance = variances["heading"]
    
    # Find the minimum length across all the data arrays
    min_length = min(len(extracted_data["northings_ground"]),
                     len(extracted_data["eastings_ground"]),
                     len(extracted_data["forward_velocity"]),
                     len(extracted_data["angular_velocity"]),
                     len(extracted_data["headings"]))

    # Trim each array to the same minimum length
    extracted_data["northings_ground"] = extracted_data["northings_ground"][:min_length]
    extracted_data["eastings_ground"] = extracted_data["eastings_ground"][:min_length]
    extracted_data["forward_velocity"] = extracted_data["forward_velocity"][:min_length]
    extracted_data["angular_velocity"] = extracted_data["angular_velocity"][:min_length]
    extracted_data["headings"] = extracted_data["headings"][:min_length]

    # Now compute the covariance matrix
    cov_northings_eastings = np.cov(extracted_data["northings_ground"], extracted_data["eastings_ground"])[0, 1]
    cov_northings_velocity = np.cov(extracted_data["northings_ground"], extracted_data["forward_velocity"])[0, 1]
    cov_eastings_velocity = np.cov(extracted_data["eastings_ground"], extracted_data["forward_velocity"])[0, 1]
    cov_northings_angular_velocity = np.cov(extracted_data["northings_ground"], extracted_data["angular_velocity"])[0, 1]
    cov_eastings_angular_velocity = np.cov(extracted_data["eastings_ground"], extracted_data["angular_velocity"])[0, 1]
    cov_velocity_angular_velocity = np.cov(extracted_data["forward_velocity"], extracted_data["angular_velocity"])[0, 1]
    cov_northings_heading = np.cov(extracted_data["northings_ground"], extracted_data["headings"])[0, 1]
    cov_eastings_heading = np.cov(extracted_data["eastings_ground"], extracted_data["headings"])[0, 1]
    cov_velocity_heading = np.cov(extracted_data["forward_velocity"], extracted_data["headings"])[0, 1]
    cov_angular_velocity_heading = np.cov(extracted_data["angular_velocity"], extracted_data["headings"])[0, 1]

    # Construct the R matrix
    R_matrix = np.array([
        [northings_variance, cov_northings_eastings, cov_northings_velocity, cov_northings_angular_velocity, cov_northings_heading],
        [cov_northings_eastings, eastings_variance, cov_eastings_velocity, cov_eastings_angular_velocity, cov_eastings_heading],
        [cov_northings_velocity, cov_eastings_velocity, forward_velocity_variance, cov_velocity_angular_velocity, cov_velocity_heading],
        [cov_northings_angular_velocity, cov_eastings_angular_velocity, cov_velocity_angular_velocity, angular_velocity_variance, cov_angular_velocity_heading],
        [cov_northings_heading, cov_eastings_heading, cov_velocity_heading, cov_angular_velocity_heading, heading_variance]
    ])

    
    return R_matrix