import numpy as np
from classification import LidarObjectClassifier
import matplotlib.pyplot as plt

# Helper function to generate a fake half-circle Lidar cluster
def generate_half_circle(radius=1.0, center=(2, 2), start_angle=0, end_angle=np.pi, noise_std=0.05, num_points=30):
    angles = np.linspace(start_angle, end_angle, num_points)
    x = center[0] + radius * np.cos(angles) + np.random.normal(0, noise_std, num_points)
    y = center[1] + radius * np.sin(angles) + np.random.normal(0, noise_std, num_points)
    return np.column_stack((x, y))

# Create the classifier
classifier = LidarObjectClassifier()

# TRAINING SET
# -------------
# Class 0: Big wide shape
cluster_0a = generate_half_circle(radius=1.5, center=(2, 2))
classifier.training_example(cluster_0a, label=0)

# Class 1: Smaller tighter shape
cluster_1a = generate_half_circle(radius=0.8, center=(-1, 3))
classifier.training_example(cluster_1a, label=1)

# Add more examples to improve generalization
cluster_0b = generate_half_circle(radius=1.4, center=(2.2, 2.1))
classifier.training_example(cluster_0b, label=0)

cluster_1b = generate_half_circle(radius=0.9, center=(-1.1, 2.8))
classifier.training_example(cluster_1b, label=1)

# Train the classifier
classifier.train_classifier()

# TESTING
# --------
# New test cluster
test_cluster = generate_half_circle(radius=0.85, center=(-1.05, 3.05))
predicted_label = classifier.classify(test_cluster)

print("Predicted label:", predicted_label)

# Optional: visualize the test
x0, y0, r, _ = classifier.fit_circle_to_points(test_cluster)
theta = np.linspace(0, 2 * np.pi, 100)
fitted_x = x0 + r * np.cos(theta)
fitted_y = y0 + r * np.sin(theta)

plt.figure(figsize=(5, 5))
plt.axis('equal')
plt.scatter(test_cluster[:, 0], test_cluster[:, 1], label='Test Lidar points', color='blue')
plt.plot(fitted_x, fitted_y, color='red', label='Fitted Circle')
plt.plot(0, 0, 'kx', label='Robot (0, 0)', markersize=10)
plt.title(f"Predicted label: {predicted_label}")
plt.legend()
plt.grid(True)
plt.show()
