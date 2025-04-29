import json

def load_log_file(filepath):
    data = []
    with open(filepath, 'r') as file:
        for line in file:
            try:
                json_object = json.loads(line)
                data.append(json_object)
            except json.JSONDecodeError:
                print("Skipping line due to JSONDecodeError.")
    return data


def inspect_lidar_data(data):
    lidar_entries = [entry for entry in data if entry['topic_name'] == '/lidar']
    
    print(f"Total /lidar entries found: {len(lidar_entries)}\n")

    for i, entry in enumerate(lidar_entries[:10]):  # Displaying first 10 for inspection
        message = entry.get('message', {})
        ranges = message.get('ranges', [])
        angles = message.get('angles', [])

        if all(value == 0 or value == 'nan' for value in ranges):
            print(f"\nEntry {i}: ALL RANGES ARE ZERO OR NaN")
        else:
            print(f"\nEntry {i}: Valid ranges detected.")

        print(f"Ranges: {ranges[:10]}...")  # Print first 10 range values for inspection
        print(f"Angles: {angles[:10]}...")  # Print first 10 angle values for inspection


# Adjust the path to your log file
filepath = "logs/all_static_corners_&_walls_20250325_135405_log.json"

log_data = load_log_file(filepath)
inspect_lidar_data(log_data)
