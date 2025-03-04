## need to add log files on git and change file_path accordingly

import json
import numpy as np

# Path to the JSON file
file_path = "C:\\Users\\danae\\Downloads\\20250304_141634_log.json"

timestamps = []
with open(file_path, "r") as file:
    for line in file:  # Read line by line for large files
        data = json.loads(line)
        if "timestamp" in data:
            timestamps.append(float(data["timestamp"]))  # Convert to float

# Sort timestamps (if not already sorted)
timestamps.sort()

# Compute time differences (dt) between consecutive timestamps
dt_values = np.diff(timestamps)  # NumPy automatically computes differences

# Print or save the results
print("Time differences (dt) between measurements:", dt_values)

# Save to a file
with open(r"C:\Users\danae\Downloads\dt_values.txt", "w") as output_file:
    for dt in dt_values:
        output_file.write(f"{dt}\n")

print(f"Computed {len(dt_values)} time differences. Saved to 'dt_values.txt'.")

# Compute and print variance of dt values
dt_variance = np.var(dt_values)
print(f"Variance of dt values: {dt_variance}")
