# testing classification.py with fake lidar data

import numpy as np
import matplotlib.pyplot as plt
from classification import LidarObjectClassifier

# Simulate Lidar points forming a half-circle arc (more realistic)
def generate_half_circle(radius=1.5, center=(2, 2), start_angle=0, end_angle=np.pi, num_points=30, noise_std=0.02):
    angles = np.linspace(start_angle, end_angle, num_points)
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    
    # Add some noise to mimic sensor error
    x += np.random.normal(0, noise_std, num_points)
    y += np.random.normal(0, noise_std, num_points)
    
    return np.column_stack((x, y))  # shape (N, 2)

# Generate the arc-shaped cluster
points = generate_half_circle()

# Use your classifier
classifier = LidarObjectClassifier()
r, d, angle, fit_error = classifier.extract_features(points)

# Print the results
print("Radius:", round(r, 3))
print("Distance to center:", round(d, 3))
print("Angle to center (rad):", round(angle, 3))
print("Fit error:", round(fit_error, 5))

# Plot the original points and the fitted circle
x0, y0, _, _ = classifier.fit_circle_to_points(points)
theta = np.linspace(0, 2 * np.pi, 100)
fitted_x = x0 + r * np.cos(theta)
fitted_y = y0 + r * np.sin(theta)

plt.figure(figsize=(5, 5))
plt.axis('equal')
plt.scatter(points[:, 0], points[:, 1], label='Lidar arc points')
plt.plot(fitted_x, fitted_y, color='red', label='Fitted full circle')
plt.plot(x0, y0, 'ro', label='Estimated center')
plt.plot(0, 0, 'kx', label='Robot position', markersize=10)
plt.title('Realistic Half-Circle Lidar Test')
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.grid(True)
plt.legend()
plt.show()
