import json
import numpy as np
import math
import matplotlib.pyplot as plt
from Libraries.new_model_feeg6043 import RangeAngleKinematics, t2v, v2t
from Libraries.new_plot_feeg6043 import plot_2dframe, show_observation
from Libraries.new_math_feeg6043 import polar2cartesian, cartesian2polar, HomogeneousTransformation, l2m, Vector,Inverse, Matrix
import copy
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessClassifier #####from any python you learn#######
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

p = Vector(3); 

p[0] = 0 #Northings
p[1] = 0 #Eastings
p[2] = 0 #Heading (rad)

x_bl = 0; y_bl = 0
t_bl = Vector(2)
t_bl[0] = x_bl
t_bl[1] = y_bl
H_bl = HomogeneousTransformation(t_bl,0)
lidar = RangeAngleKinematics(x_bl, y_bl, distance_range = [0.1, 1], scan_fov = np.deg2rad(60), n_beams = 30)


def fit_line_to_points(points,fit_error_tolerance = 0.5 ):
    model = LinearRegression().fit(points[:, 0].reshape(-1, 1), points[:, 1])
    predictions = model.predict(points[:, 0].reshape(-1, 1))

    fit_error = np.mean(np.abs(predictions - points[:, 1])) 
    if fit_error < fit_error_tolerance:
        return fit_error
    else:
        return None


def fit_circle_to_points(points, fit_error_tolerance=0.5):
    """
    Fits a circle directly using polar coordinates [r, theta].
    
    Parameters:
    - points: N x 2 numpy array where each row is [r, theta].
    - fit_error_tolerance: Maximum acceptable fit error (optional).
    
    Returns:
    - r0: Estimated radius of the circle.
    - theta0: Estimated angle of the circle's center (in radians).
    - radius: Estimated radius of the circle.
    """

    # Check if the input is a GPC_input_output object
    if isinstance(points, GPC_input_output):
        points = points.data_filled  # Extract the actual data if it's wrapped in an object

    # Extract r and theta from the polar coordinates
    r = points[:, 0]
    theta = points[:, 1]

    # Convert Polar to Cartesian coordinates internally for calculation
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    # Circle fitting using least squares method (same as your original code)
    A = np.c_[2 * x, 2 * y, np.ones(x.shape)]
    B = x**2 + y**2

    # Solving for circle parameters
    C = np.linalg.lstsq(A, B, rcond=None)[0]
    x0, y0, c = C

    # Convert the center (x0, y0) back to polar coordinates
    r0 = np.sqrt(x0**2 + y0**2)
    theta0 = np.arctan2(y0, x0)
    radius = np.sqrt(x0**2 + y0**2 + c)

    # Calculate fit error directly in polar space
    estimated_r = np.sqrt((x - x0)**2 + (y - y0)**2)
    fit_error = np.sqrt(np.mean((estimated_r - radius)**2))

    # Check if the fit error is within the tolerance
    if fit_error < fit_error_tolerance:
        return r0, theta0, radius
    else:
        return None, None, None


def show_scan(p_eb, lidar, observations, show_lines = True):
    """ Plots observations, field of view and robot pose
    """

    ######################## Calculate FOV    
    range_max = lidar.distance_range[1]
    range_min = lidar.distance_range[0]    
    fov = lidar.scan_fov
        
    r_ = []
    theta_ = []    
    
    # for field of view
    theta = np.linspace(-fov / 2, fov / 2, 30)    
        
    for i in theta:
        r_.append(range_max)
        theta_.append(i)        
    for i in reversed(theta):
        r_.append(range_min)
        theta_.append(i)    
    r_.append(range_max)
    theta_.append(-fov/2)

    fov = l2m([r_,theta_])
    ######################## Plot the FOV        
    
    t_lm = Vector(2) # lidar frame measurement placeholder    
    t_em = Vector(2) # environment frame measurement
    
    fov_x = []
    fov_y = []

    H_eb = HomogeneousTransformation(p_eb[0:2],p_eb[2])
        
    for z_fov in fov:    
        t_lm[0],t_lm[1] = polar2cartesian(z_fov[0],z_fov[1])      
        t_em = t2v((H_eb.H@lidar.H_bl.H)@v2t(t_lm))
    
        fov_x.append(t_em[0])
        fov_y.append(t_em[1])  
        
    if show_lines == True: plt.plot(fov_y, fov_x,'orange')

    if len(observations) != 0:            
        for z_lm in observations:    
            t_lm[0],t_lm[1] = polar2cartesian(z_lm[0],z_lm[1])
            show_observation(H_eb,t2v(lidar.H_bl.H@v2t(t_lm)),Matrix(2,2),None,ax, show_lines)        
        
    else:        
        cf=plot_2dframe(['pose','b','b'],[H_eb.H,H_eb.H],False,False)
        
    plt.xlabel('Eastings, m')
    plt.ylabel('Northings, m')
    plt.axis('equal')

def show_scan(p_eb, lidar, observations, show_lines = True):
    """ Plots observations, field of view and robot pose
    """

    ######################## Calculate FOV    
    range_max = lidar.distance_range[1]
    range_min = lidar.distance_range[0]    
    fov = lidar.scan_fov
        
    r_ = []
    theta_ = []    
    
    # for field of view
    theta = np.linspace(-fov / 2, fov / 2, 30)    
        
    for i in theta:
        r_.append(range_max)
        theta_.append(i)        
    for i in reversed(theta):
        r_.append(range_min)
        theta_.append(i)    
    r_.append(range_max)
    theta_.append(-fov/2)

    fov = l2m([r_,theta_])
    ######################## Plot the FOV        
    
    t_lm = Vector(2) # lidar frame measurement placeholder    
    t_em = Vector(2) # environment frame measurement
    
    fov_x = []
    fov_y = []

    H_eb = HomogeneousTransformation(p_eb[0:2],p_eb[2])
        
    for z_fov in fov:    
        t_lm[0],t_lm[1] = polar2cartesian(z_fov[0],z_fov[1])      
        t_em = t2v((H_eb.H@lidar.H_bl.H)@v2t(t_lm))
    
        fov_x.append(t_em[0])
        fov_y.append(t_em[1])  
        
    if show_lines == True: plt.plot(fov_y, fov_x,'orange')

    if len(observations) != 0:            
        for z_lm in observations:    
            t_lm[0],t_lm[1] = polar2cartesian(z_lm[0],z_lm[1])
            show_observation(H_eb,t2v(lidar.H_bl.H@v2t(t_lm)),Matrix(2,2),None,ax, show_lines)        
        
    else:        
        cf=plot_2dframe(['pose','b','b'],[H_eb.H,H_eb.H],False,False)
        
    plt.xlabel('Eastings, m')
    plt.ylabel('Northings, m')
    plt.axis('equal')

def find_corner(corner, threshold = 0.01):
    # identify the reference coordinate as the inflection point
    
    # Step 1: Compute slope
    slope = np.gradient(corner[:, 0])
    
    # Step 2: Compute the second derivative (curvature)
    curvature = np.gradient(slope)
    
    # Step 3: Check if criteria is more than threshold    
    print('Max inflection value is ',np.nanmax(abs(np.gradient(np.gradient(curvature)))), ': Threshold ',threshold)
    if np.nanmax(abs(np.gradient(np.gradient(curvature)))) > threshold:
        # compute index of inflection point    
        largest_inflection_idx = np.nanargmax(abs(np.gradient(np.gradient(curvature))))  ####wheres the larges inflection point#### for finding corners
        
        r = corner[largest_inflection_idx, 0]  # Radial distance at the largest curvature
        theta = corner[largest_inflection_idx, 1]  # Angle at the largest curvature
        return r, theta, largest_inflection_idx
    
    else:
        return None, None, None  # No inflection points found

class GPC_input_output:
    def __init__(self, data, label):
        """
        Initializes an observation with data and a label.

        Parameters:
        data (matrix): The observation data (e.g., a matrix).
        data_filled (matrix): The observation data after zero offset and making nan's mean
        label (str): The label associated with the observation.
        ne_representative: representative northings and eastings location
        """
        self.data = data
        self.data_filled = self._fill_nan(data)
        self.label = label
        self.ne_representative = None 
        # make filled and zero offset version        
        
    def _fill_nan(self,data):
        data_filled = np.copy(data)
        mean=np.nanmean(data[:,0])  
        for i in range(len(data[:,1])):
            if np.isnan(data[i,0]):
                data_filled[i,0]=0
            else: 
                data_filled[i,0]=data[i,0]-mean
        return data_filled



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
            if isinstance(data, list):
                extracted_data.append(np.array(data))  # Convert to numpy array for consistency
            elif isinstance(data, dict):  # If it's a dict, extract values as tuple
                extracted_data.append(np.array(tuple(data.get(k, np.nan) for k in data)))
            else:  # If it's a single value, just append it
                extracted_data.append(np.array([data]))

        return extracted_data

    
#"logs/all_static_corners_&_walls_20250325_135405_log.json"


def format_scan(filepath, threshold = 0.001, fit_error_tolerance = 0.05, fit_error_tolerance_wall = 0.5):

    variables = ImportLog(filepath)
    r = variables.extract_data("/lidar", ["message", "ranges"])
    theta = variables.extract_data("/lidar", ["message", "angles"])
    timestamps = variables.extract_data("/groundtruth", ["timestamp"])

    ###########________initilise corner_training_______###########

    if not isinstance(r, list):
        raise TypeError("Expected r to be a list, got {} instead.".format(type(r)))
    if not isinstance(theta, list):
        raise TypeError("Expected theta to be a list, got {} instead.".format(type(theta)))

    if len(r) == 0 or len(theta) == 0:
        raise ValueError("Extracted r or theta is empty. Ensure your logs have valid data.")

    if isinstance(r[0], list) or isinstance(r[0], np.ndarray):
        r[0] = np.array(r[0])
    else:
        r = [np.array(r)]  # Wrap in a list if it is a single array

    if isinstance(theta[0], list) or isinstance(theta[0], np.ndarray):
        theta[0] = np.array(theta[0])
    else:
        theta = [np.array(theta)]  # Wrap in a list if it is a single array

    observation = np.column_stack((r[0], theta[0]))  # Properly format the data
    corner_example = GPC_input_output(observation, None)
    corner_training = [corner_example]

    for i in range(len(r)):
        if isinstance(r[i], list) or isinstance(r[i], np.ndarray):
            r[i] = np.array(r[i])
        else:
            print(f"Warning: Unexpected type for r[{i}]. Skipping.")
            continue

        if isinstance(theta[i], list) or isinstance(theta[i], np.ndarray):
            theta[i] = np.array(theta[i])
        else:
            print(f"Warning: Unexpected type for theta[{i}]. Skipping.")
            continue

        observation = np.column_stack((r[i], theta[i]))  # Ensure column stacking works
        z_lm = Vector(2)

        ##z_lm[0] = r, z_lm[1]= theta, loc  = its index within the scan 
        z_lm[0], z_lm[1], loc = find_corner(observation, threshold)
        print(z_lm, loc)

        new_observation = GPC_input_output(observation, None)

        if loc is not None:
            new_observation.label = 'corner'
            new_observation.ne_representative = z_lm
            print('Map observation made at, Northings = ', new_observation.ne_representative[0], 'm, Eastings =', new_observation.ne_representative[1], 'm')  

            corner_training.append(new_observation)

        else: 
            z_lm[0], z_lm[1], r_val = fit_circle_to_points(new_observation, fit_error_tolerance)

            if r_val is not None:
                new_observation.label = 'object'
                new_observation.ne_representative = z_lm
                print('Map observation made at, Northings = ', new_observation.ne_representative[0], 'm, Eastings =', new_observation.ne_representative[1], 'm')  

                corner_training.append(new_observation)

            else:
                error = fit_line_to_points(new_observation, fit_error_tolerance_wall)

                if error is not None:
                    new_observation.label = 'Wall'
                    new_observation.ne_representative = z_lm
                    print('Map observation made at, Northings = ', new_observation.ne_representative[0], 'm, Eastings =', new_observation.ne_representative[1], 'm')  

                    corner_training.append(new_observation)

    for i in range(len(corner_training)):
        print('Entry:', i, ', Class', corner_training[i].label, ', Size', corner_training[i].data_filled[:, 0].size)
        print('Data type: Radius', corner_training[i].data_filled[:, 0])
        print('Data type:Theta', corner_training[i].data_filled[:, 1])


format_scan("logs/all_static_corners_&_walls_20250325_135405_log.json")
format_scan("logs/all_static_corners_&_walls_20250325_135405_log.json")
