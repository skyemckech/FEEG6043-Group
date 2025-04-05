import json
import numpy as np
import math
import matplotlib.pyplot as plt

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
    
logFile = ImportLog("logs/20250404_114101_log.json")
p_ref_0 = logFile.extract_data("/p_ref",["message","vector","x"])
p_ref_1 = logFile.extract_data("/p_ref",["message","vector","y"])
print(type(p_ref_0))

p_ref_0 = [x - p_ref_0[0] for x in p_ref_0]
p_ref_1 = [x - p_ref_1[0] for x in p_ref_1]
p_ref_0 = p_ref_0[::2]
p_ref_1 = p_ref_1[::2]


dx = np.diff(p_ref_0)
dy = np.diff(p_ref_1)

lapx = [0,1.4,1.4,0.3,0.3,1.1,1.1,0]
lapy = [0,0,1.4,1.4,0.3,0.3,1.1,1.1]

plt.figure(figsize=(6, 4))
plt.xlim(0, 1.75)  # Set x-axis limits
plt.ylim(0, 1.75)  # Set y-axis limits
plt.quiver(p_ref_1[:-1], p_ref_0[:-1],dy, dx, angles='xy', scale_units='xy', scale=0.5, color='blue')
plt.scatter(lapy,lapx, color = 'red', s=80, label = "Waypoint")
for i, (x, y) in enumerate(zip(lapy, lapx)):
    plt.text(x+0.05, y+0.05, str(i), fontsize=20, color='red')  # Adjust offsets as needed



plt.plot(lapy[0], lapx[0], 'o', label='Start & End',  markersize=20)  # Add the first point
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12, labelspacing=1.5)
plt.xlabel("Eastings E (m)", size = 20)
plt.ylabel("Northings N (m)", size = 20)
plt.title("Reference trajectory for all tests", size = 30)
plt.show()
