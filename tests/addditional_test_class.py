import numpy as np
import matplotlib.pyplot as plt
from classification import LidarObjectClassifier  # Make sure classifier.py is in the same folder

# ----------------------------------------
# Function to generate a half-circle cluster
# ----------------------------------------
def make_cluster(radius, center, label=None, noise=0.05, start_angle=0, end_angle=np.pi, num_points=30):
    angles = np.linspace(start_angle, end_angle, num_points)
    x = center[0] + radius * np.cos(angles) + np.random.normal(0, noise, num_points)
    y = center[1] + radius * np.sin(angles) + np.random.normal(0, noise, num_points)
    points = np.column_stack((x, y))
    return points, label

# ----------------------------------------
# Create the classifier and training data
# ----------------------------------------
clf = LidarObjectClassifier()

# Class 0: object (e.g., pole)
cluster_0a, _ = make_cluster(0.6, (1.5, 2.5))
clf.training_example(cluster_0a, label=0)

cluster_0b, _ = make_cluster(0.7, (1.7, 2.3))
clf.training_example(cluster_0b, label=0)

# Class 1: wall (flatter shape)
cluster_1a, _ = make_cluster(5.0, (0, 4))
clf.training_example(cluster_1a, label=1)

cluster_1b, _ = make_cluster(6.0, (0.5, 3.5))
clf.training_example(cluster_1b, label=1)

# Class 2: corner (incomplete arc or irregular shape)
cluster_2a, _ = make_cluster(1.2, (2, 2), start_angle=0, end_angle=np.pi/2)
clf.training_example(cluster_2a, label=2)

cluster_2b, _ = make_cluster(1.0, (2.5, 1.8), start_angle=0, end_angle=np.pi/3)
clf.training_example(cluster_2b, label=2)

# ----------------------------------------
# Train the classifier
# ----------------------------------------
clf.train_classifier()

# ----------------------------------------
# Test on a new cluster
# ----------------------------------------
test_points, _ = make_cluster(0.65, (1.6, 2.6))  # Simulate unknown object
prediction = clf.classify(test_points)

# Get features for annotation
r, dist, angle, fit_error = clf.extract_features(test_points)

# ----------------------------------------
# Plot the cluster and fitted circle
# ----------------------------------------
x0, y0, _, _ = clf.fit_circle_to_points(test_points)
theta = np.linspace(0, 2 * np.pi, 100)
circle_x = x0 + r * np.cos(theta)
circle_y = y0 + r * np.sin(theta)

plt.figure(figsize=(5, 5))
plt.axis('equal')
plt.scatter(test_points[:, 0], test_points[:, 1], label='Test points', color='blue')
plt.plot(circle_x, circle_y, color='red', label='Fitted circle')
plt.plot(0, 0, 'kx', label='Robot (0,0)', markersize=10)
plt.title(f"Prediction: Class {prediction}")
plt.xlabel("x (meters)")
plt.ylabel("y (meters)")
plt.grid(True)
plt.legend()

# --- Annotate with extracted features ---
text = f"Radius: {r:.3f}\nDistance: {dist:.3f}\nAngle: {angle:.3f} rad\nFit error: {fit_error:.5f}"
plt.text(0.05, 0.95, text, transform=plt.gca().transAxes,
         verticalalignment='top', fontsize=9,
         bbox=dict(boxstyle="round", facecolor='white', edgecolor='gray'))

plt.show()
