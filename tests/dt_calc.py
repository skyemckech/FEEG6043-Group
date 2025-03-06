import json
import numpy as np

# Path to the JSON file
file_path = r"C:/Users/danae/Folder/FEEG6043-Group-2/logs/20250303_190632_log.json"

# Initialize a dictionary to store timestamps by topic
timestamps_by_topic = {
    "/groundtruth": [],
    "/true_wheel_speeds": [],
    "/lidar": [],
    "/aruco": [],
    "/est_pose": [],
    "/wheel_speeds_cmd": []
}

# Read the log file and collect timestamps by topic
with open(file_path, "r") as file:
    for line in file:  # Read line by line for large files
        data = json.loads(line)
        topic = data.get("topic_name")
        if "timestamp" in data and topic in timestamps_by_topic:
            timestamps_by_topic[topic].append(float(data["timestamp"]))

# Sort and compute dt values for each topic
dt_values_by_topic = {}
for topic, timestamps in timestamps_by_topic.items():
    timestamps.sort()
    dt_values = np.diff(timestamps) if len(timestamps) > 1 else []
    dt_values_by_topic[topic] = dt_values

# Save dt values to a file for each topic
output_path = r"C:/Users/danae/Downloads/"
for topic, dt_values in dt_values_by_topic.items():
    filename = f"dt_values_{topic.replace('/', '_')}.txt"
    with open(output_path + filename, "w") as output_file:
        for dt in dt_values:
            output_file.write(f"{dt:.4f}\n")  # Limit to 4 decimal places
    print(f"Saved dt values for {topic} to {filename}")

# Calculate and display variance and mean of dt values for each topic
print("\nSummary of dt values by topic:")
for topic, dt_values in dt_values_by_topic.items():
    if len(dt_values) > 0:  # Correct check for non-empty arrays
        dt_variance = np.var(dt_values)
        dt_mean = np.mean(dt_values)
        print(f"{topic} - Variance of dt: {dt_variance:.4f}, Mean dt: {dt_mean:.4f}")
    else:
        print(f"{topic} - Not enough data to compute variance and mean.")
