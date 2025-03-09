# Note: 
# This script is used to interpolate the variance data of the logs. The variance data is calculated for each log file and then interpolated to align all data on common timestamps. The interpolated variance data is then plotted to visualize the variance of the different log data over time.
# 	Mean Dt
# groundtruth 	0.0925
# true_wheel_speeds	0.0915
# lidar	1.0211
# aruco	1.0192
# est_pose	0.0996
# wheel_speeds_cmd	0.0996


import json
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os

# Function to load and extract relevant data from JSON logs
class ImportLog:
    def __init__(self, filepath):
        self.filepath = filepath  # Store the file path of the log
        self.data = self._load_data()  # Load and process the data immediately

    def _load_data(self):
        # Initialize a dictionary to store relevant data from the log
        data = {'timestamps': [], 'northings': [], 'eastings': [], 'headings': [], 'wheel_right': [], 'wheel_left': []}
        with open(self.filepath, 'r') as file:
            for line in file:
                try:
                    entry = json.loads(line.strip())  # Parse each line as a JSON object
                    # Extract data from the '/groundtruth' topic
                    if entry['topic_name'] == '/groundtruth':
                        data['timestamps'].append(float(entry['timestamp']))
                        data['northings'].append(entry['message']['position']['x'])
                        data['eastings'].append(entry['message']['position']['y'])
                        data['headings'].append(entry['message']['orientation']['z'])
                    # Extract data from the '/true_wheel_speeds' topic
                    elif entry['topic_name'] == '/true_wheel_speeds':
                        # Append wheel speed data while ensuring alignment with timestamps
                        if len(data['wheel_right']) < len(data['timestamps']):
                            data['wheel_right'].append(entry['message']['vector']['x'])
                        if len(data['wheel_left']) < len(data['timestamps']):
                            data['wheel_left'].append(entry['message']['vector']['y'])
                except json.JSONDecodeError:
                    continue
        # Truncate all lists to the same minimum length to avoid inconsistencies
        min_length = min(len(data['timestamps']), len(data['northings']), len(data['eastings']), len(data['headings']), len(data['wheel_right']), len(data['wheel_left']))
        for key in data.keys():
            data[key] = data[key][:min_length]
        # Normalize timestamps to start at 0
        start_time = data['timestamps'][0]
        data['timestamps'] = [t - start_time for t in data['timestamps']]
        return data

# Function to perform linear interpolation for each log
def interpolate_logs(logs, common_timestamps):
    interpolated_logs = []
    for log in logs:
        interpolated_data = {'timestamps': common_timestamps}
        for key in ['northings', 'eastings', 'headings', 'wheel_right', 'wheel_left']:
            if len(log['timestamps']) > 1:  # Only interpolate if there are enough data points
                f = interp1d(log['timestamps'], log[key], kind='linear', fill_value='extrapolate')
                interpolated_data[key] = f(common_timestamps)
            else:
                interpolated_data[key] = np.full(len(common_timestamps), np.nan)
        interpolated_logs.append(interpolated_data)
    return interpolated_logs

# Load data from all log files in the correct directory
log_directory = 'C:\Users\conno\OneDrive\Desktop\5th Year\IMB\FEEG6043-Group-1\logs\'
filepaths = [os.path.join(log_directory, f) for f in [
    '0.1_Scale.json',
    '0.2_Scale.json'
   ]]
original_logs = [ImportLog(fp).data for fp in filepaths]

# Generate a common set of timestamps for interpolation
min_time = max(min(log['timestamps']) for log in original_logs)
max_time = min(max(log['timestamps']) for log in original_logs)
dt = 0.5  # Define the interpolation time step
common_timestamps = np.arange(min_time, max_time, dt)

# Perform interpolation of logs to align all data on common timestamps
interpolated_logs = interpolate_logs(original_logs, common_timestamps)

# Function to pad data arrays with NaN to ensure consistent array shapes
def pad_data_arrays(logs, key, target_length):
    padded_data = []
    for log in logs:
        data = log[key]
        if len(data) < target_length:
            data = np.pad(data, (0, target_length - len(data)), constant_values=np.nan)
        padded_data.append(data)
    return np.array(padded_data)

# Plot the original variance data
plt.figure(figsize=(15, 10))
for i, key in enumerate(['northings', 'eastings', 'headings', 'wheel_right', 'wheel_left']):
    plt.subplot(2, 3, i + 1)
    target_length = max(len(log[key]) for log in original_logs)
    data_array = pad_data_arrays(original_logs, key, target_length)
    variance = np.nanvar(data_array, axis=0)  # Compute variance, ignoring NaNs
    plt.plot(original_logs[0]['timestamps'][:len(variance)], variance, label=f'Original Variance of {key}', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Variance')
    plt.title(f'Original Variance of {key.capitalize()} over Time')
    plt.legend()

plt.tight_layout()
plt.show()
plt.close()

# Plot the interpolated variance data
plt.figure(figsize=(15, 10))
for i, key in enumerate(['northings', 'eastings', 'headings', 'wheel_right', 'wheel_left']):
    plt.subplot(2, 3, i + 1)
    data_array = np.array([log[key] for log in interpolated_logs])
    variance = np.nanvar(data_array, axis=0)  # Compute the variance across all logs
    plt.plot(common_timestamps, variance, label=f'Interpolated Variance of {key}', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Variance')
    plt.title(f'Interpolated Variance of {key.capitalize()} over Time')
    plt.legend()

plt.tight_layout()
plt.show()
plt.close()
