import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from Libraries.new_model_feeg6043 import RangeAngleKinematics, t2v, v2t
from Libraries.new_plot_feeg6043 import plot_2dframe, show_observation
from Libraries.new_math_feeg6043 import polar2cartesian, cartesian2polar, HomogeneousTransformation, l2m, Vector, Inverse, Matrix
import matplotlib.pyplot as plt
import copy
from sklearn.decomposition import PCA


class LidarObjectClassifier:
    # This class processes Lidar points and extracts features from them

    def __init__(self):
        # Stores training data (features) and their corresponding labels
        self.X_train = []
        self.y_train = []

    def fit_circle_to_points(self, points):
        # Fit a circle to the given 2D points using the least squares method
        x = points[:, 0]
        y = points[:, 1]

        A = np.c_[2 * x, 2 * y, np.ones(x.shape)]
        B = x**2 + y**2

        C = np.linalg.lstsq(A, B, rcond=None)[0]
        x0, y0, c = C
        r = np.sqrt(x0**2 + y0**2 + c)

        distances = np.sqrt((x - x0)**2 + (y - y0)**2)
        fit_error = np.sqrt(np.mean((distances - r)**2))

        return x0, y0, r, fit_error

    # def fit_line_to_points(self, points):
    #     # Fit a line to the given 2D points using PCA
    #     pca = PCA(n_components=1)
    #     pca.fit(points)
    #     line_fit_error = np.mean(np.abs(pca.transform(points)))
    #     return line_fit_error

    def fit_line_to_points(self, points):
        model = LinearRegression().fit(points[:, 0].reshape(-1, 1), points[:, 1])
        predictions = model.predict(points[:, 0].reshape(-1, 1))

        # Calculate MSE (Mean Squared Error)
        mse = mean_squared_error(points[:, 1], predictions)

        # Calculate R² Score
        r2 = r2_score(points[:, 1], predictions)

        # Line fit error is now a combination of MSE and R²
        line_fit_error = mse / (1 - r2 + 1e-6)  # Adding a small number to prevent division by zero
        return line_fit_error

    def find_corner(self, points, threshold=0.01):
        # Identify a corner by detecting inflection points using curvature analysis
        slope = np.gradient(points[:, 0])
        curvature = np.gradient(slope)
        max_inflection = np.nanmax(abs(np.gradient(np.gradient(curvature))))

        if max_inflection > threshold:
            largest_inflection_idx = np.nanargmax(abs(np.gradient(np.gradient(curvature))))
            r = points[largest_inflection_idx, 0]
            theta = points[largest_inflection_idx, 1]
            return r, theta, largest_inflection_idx, max_inflection
        else:
            return None, None, None, 0

    def extract_features(self, points):
        # Extract features for each type of classification
        x0, y0, r, fit_error = self.fit_circle_to_points(points)
        line_fit_error = self.fit_line_to_points(points)
        _, _, _, corner_value = self.find_corner(points)

        distance = np.sqrt(x0**2 + y0**2)
        angle = np.arctan2(y0, x0)

        return [r, distance, angle, fit_error, line_fit_error, corner_value]

    def training_example(self, points, label):
        # Add a labeled training example
        features = self.extract_features(points)
        self.X_train.append(features)
        self.y_train.append(label)

    def train_classifier(self):
        # Train the Gaussian Process Classifier
        X = np.array(self.X_train)
        y = np.array(self.y_train)
        kernel = 1.0 * RBF(1.0)
        self.classifier = GaussianProcessClassifier(kernel=kernel)
        self.classifier.fit(X, y)

    def classify(self, points):
        # Classify new points using the trained classifier
        features = self.extract_features(points)
        label = self.classifier.predict([features])[0]
        return label


def polar_to_cartesian(polar_data):
    # Convert polar data to Cartesian (x, y)
    cartesian_points = []
    for r, theta in polar_data:
        if not np.isnan(r):
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            cartesian_points.append([x, y])
    return np.array(cartesian_points)


classifier = LidarObjectClassifier()

# Example Lidar data in polar coordinates (r, theta) format
lidar_data = np.array([
    [0.2595, -1.04474318],
    [0.2585, -1.04447047],
    [0.2600, -1.02619906],
    [0.2600, -1.01174555],
    [0.2610, -0.98883810],
])

# Convert to Cartesian
cartesian_points = polar_to_cartesian(lidar_data)

# Label the points as a wall for training
classifier.training_example(cartesian_points, 'wall')

# Train the classifier
classifier.train_classifier()

# Classify a new set of points
label = classifier.classify(cartesian_points)
print(f'Classified as: {label}')
