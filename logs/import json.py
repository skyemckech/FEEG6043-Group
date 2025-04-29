import json
import matplotlib.pyplot as plt

def extract_measured_est_distance_x(file_paths):
    """
    Extract the x values from the /measured_est_distance topic in multiple JSONL files.

    Args:
        file_paths (list): List of file paths to JSONL files.

    Returns:
        list: A list of x values under /measured_est_distance.
    """
    x_values = []
    timestamps = []

    for file_path in file_paths:
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    if record.get("topic_name") == "/measured_est_distance":
                        x_value = record["message"]["vector"]["x"]
                        x_values.append(x_value)
                        timestamp = record
                except (json.JSONDecodeError, KeyError):
                    # Ignore lines that are not valid JSON or do not contain expected structure
                    continue

    return x_values

# File paths to the 4 JSONL files
file_paths = [
    "file1.jsonl",
    "file2.jsonl",
    "file3.jsonl",
    "file4.jsonl"
]

# Extract x values
x_values = extract_measured_est_distance_x(file_paths)

# Print the result
print("Extracted x values:", x_values)

x_values0, timestamps0 = extract_x_values_and_timestamps("logs/20250309_200252_log.json")
# x_values1, timestamps1 = extract_x_values_and_timestamps("logs/20250309_202150_log.json")
# x_values2, timestamps2 = extract_x_values_and_timestamps("logs/20250309_202326_log.json")
# x_values3, timestamps3 = extract_x_values_and_timestamps("logs/20250309_202513_log.json")

plt.plot(x_values0, timestamps0)
plt.show()