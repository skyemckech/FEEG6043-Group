# just an attempt to train the model with some data


# training_data.py

import numpy as np
from combined_classification import LidarObjectClassifier, polar_to_cartesian

classifier = LidarObjectClassifier()

# 1. Wall Training Data
wall_data = np.array([
    [0.2, -0.5],
    [0.21, -0.48],
    [0.22, -0.46],
    [0.23, -0.44],
    [0.24, -0.42],
])
wall_cartesian = polar_to_cartesian(wall_data)
classifier.training_example(wall_cartesian, 'wall')


# 2. Corner Training Data
corner_data = np.array([
    [0.3, -0.7],
    [0.29, -0.65],
    [0.25, -0.6],
    [0.2, -0.55],
    [0.15, -0.5],
])
corner_cartesian = polar_to_cartesian(corner_data)
classifier.training_example(corner_cartesian, 'corner')


# 3. Round Object Training Data
round_object_data = np.array([
    [0.3, -0.9],
    [0.31, -0.87],
    [0.32, -0.85],
    [0.31, -0.82],
    [0.3, -0.8],
])
round_object_cartesian = polar_to_cartesian(round_object_data)
classifier.training_example(round_object_cartesian, 'round object')


# Train the Classifier with the labeled data
classifier.train_classifier()

# Save the trained model for reuse if desired (optional)
import pickle
with open('trained_classifier.pkl', 'wb') as file:
    pickle.dump(classifier, file)

print("Training completed successfully.")
