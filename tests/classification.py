import numpy as np      

class LidarObjectClassifier:
    # This class processes Lidar points and extracts features from them

    def __init__(self):
        # where data will be stored
        self.X_train = []
        self.y_train = []

    def fit_circle_to_points(self, points):
        # fit a circle to the points using lidar 2D points 
        x = points[:, 0]
        y = points[:, 1]    

        A = []
        B = []

        # Loop through every (x, y) point to build A and B
        for i in range(len(x)):
            xi = x[i]
            yi = y[i]
            A.append([2 * xi, 2 * yi, 1])            # 2xi, 2yi, 1
            B.append(xi**2 + yi**2)                  # xi^2 + yi^2

        # Convert A and B to NumPy arrays
        A = np.array(A)
        B = np.array(B)

        # Solve the least squares system A * [x0, y0, c] = B
        C = np.linalg.lstsq(A, B, rcond=None)[0]

        x0 = C[0]
        y0 = C[1]
        c  = C[2]

        # Compute radius
        r = np.sqrt(x0**2 + y0**2 + c)

        #Compute fit error: how close points are to the fitted circle
        distances = np.sqrt((x - x0)**2 + (y - y0)**2)
        fit_error = np.sqrt(np.mean((distances - r)**2))  # Root Mean Square Error

        return x0, y0, r, fit_error  # Return center, radius, and fit quality

    def extract_features(self, points):
        # Extract features from the points
        x0, y0, r, fit_error = self.fit_circle_to_points(points)

        # Calculate distance from robot to center of circle
        distance = np.sqrt(x0**2 + y0**2)

        # Calculate angle from robot to center of circle
        angle = np.arctan2(y0, x0)

        return r, distance, angle, fit_error
    
#training example

def training_example(self,points,label):
        # Extract features from the points
        r, d, angle, fit_error = self.extract_features(points)
        # Store the features and label
        self.X_train.append([r, d, angle, fit_error])
        self.y_train.append(label)

        