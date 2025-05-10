import json
import pandas as pd
import numpy as np
def parse_json_file(file_path):
    data_by_topic = {}  # Dictionary to store data organised by topic

    with open(file_path, 'r') as file:
        for line in file:
            try:
                # Parse each JSON object
                obj = json.loads(line.strip())

                # Organise data by topic
                obj_topic = obj.get("topic", "Unknown")
                if obj_topic not in data_by_topic:
                    data_by_topic[obj_topic] = []
                data_by_topic[obj_topic].append(obj)

            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")

    return data_by_topic    

class myClassifier():
    def __init__(self):
        self.file_path = [(r'logs/circle_0d_0mm'),
            ]           # Replace with your file path
        
    def pull_logs(self):
        for path in self.file_path:
            data_by_topic = parse_json_file(path)
            print(data_by_topic)

gamer = myClassifier()
gamer.pull_logs()