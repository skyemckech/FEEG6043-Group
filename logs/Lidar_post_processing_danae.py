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
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.base import clone
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
import numpy as np

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

m_x = []
m_y = []

for x in np.arange(-1, 1, 0.01):
    m_x.append(x)
    m_y.append(-1) #west wall      
for x in np.arange(-1, 1, 0.01):
    m_x.append(x)
    m_y.append(1) #east wall
for y in np.arange(-1, 1, 0.01):
    m_x.append(-1)
    m_y.append(y) #south wall
for y in np.arange(-1, 1, 0.01):
    m_x.append(1)
    m_y.append(y) #north wall

environment_map = l2m([m_x,m_y])
H_eb = HomogeneousTransformation(p[0:2],p[2])
H_el = HomogeneousTransformation()
H_el.H = H_eb.H@lidar.H_bl.H
fig,ax = plt.subplots()


def fit_line_to_points(points,fit_error_tolerance = 0.5 ):
    model = LinearRegression().fit(points[:, 0].reshape(-1, 1), points[:, 1])
    predictions = model.predict(points[:, 0].reshape(-1, 1))

    fit_error = np.mean(np.abs(predictions - points[:, 1])) 
    if fit_error < fit_error_tolerance:
        return fit_error
    else:
        return None


def fit_circle_to_points(points, fit_error_tolerance=0.005):
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
    x = np.zeros_like(r)
    y = np.zeros_like(r)

    # Convert Polar to Cartesian coordinates internally for calculation
    for i in range(len(r)):
        x[i] = r[i] * np.cos(theta[i])
        y[i] = r[1] * np.sin(theta[i])

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
        return r0, theta0, radius, fit_error
    else:
        return None, None, None, None


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
    #print('Max inflection value is ',np.nanmax(abs(np.gradient(np.gradient(curvature)))), ': Threshold ',threshold)
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


def format_scan_corner(filepath, threshold = 0.001, fit_error_tolerance = 0.01, fit_error_tolerance_wall = 0.005):
    fit_error = None
    variables = ImportLog(filepath)
    r = variables.extract_data("/lidar", ["message", "ranges"])
    theta = variables.extract_data("/lidar", ["message", "angles"])
    timestamps = variables.extract_data("/groundtruth", ["timestamp"])
    j = 0

    ###########________random bull shit which fixes formating_______###########

    if not isinstance(r, list):
        raise TypeError("Expected r to be a list, got {} instead.".format(type(r)))
    if not isinstance(theta, list):
        raise TypeError("Expected theta to be a list, got {} instead.".format(type(theta)))

    if len(r) == 0 or len(theta) == 0:
        raise ValueError("Extracted r or theta is empty. Ensure your logs have valid data.")

    if isinstance(r[0], list) or isinstance(r[0], np.ndarray):
        r[0] = np.array(r[0])
    else:
        r = [np.array(r)]  

    if isinstance(theta[0], list) or isinstance(theta[0], np.ndarray):
        theta[0] = np.array(theta[0])
    else:
        theta = [np.array(theta)]  

    observation = np.column_stack((r[0], theta[0]))  
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

        observation = np.column_stack((r[i], theta[i]))  
        z_lm = Vector(2)

        z_lm[0], z_lm[1], loc = find_corner(observation, threshold)
        #print(z_lm, loc)

        new_observation = GPC_input_output(observation, None)
        values = new_observation.data
        #print(values)

        #values = values.astype(float)
        #print("Values[0]:",values[:,0],"Values[1]:",values[:,1],)

        ####plotting funcitons####
        
        # if j < 6:
        #     j = j + 1
        #     fig,ax = plt.subplots()
        #     show_scan(p, lidar, observation)
        #     ax.scatter(m_y, m_x,s=0.01)
        #     plt.title(loc)
        #     plt.show()
        #     print(j)
        # else:
        #     print("yummy cummy")

            ####if the number of nan's is more than X of the total size then pass#######
        if np.count_nonzero(~np.isnan(values[:,0])) > 0.1 * len(values[:,0]):

            if loc is not None:
                new_observation.label = 'corner'
                new_observation.ne_representative = z_lm
                #print('Map observation made at, Northings = ', new_observation.ne_representative[0], 'm, Eastings =', new_observation.ne_representative[1], 'm')  
                print("corner")

                corner_training.append(new_observation)

            else: 
                z_lm[0], z_lm[1], r_val, fit_error = fit_circle_to_points(new_observation, fit_error_tolerance)
                print("object fit error is here::::::--------",fit_error)

                if r_val is not None:
                    new_observation.label = 'not corner'
                    new_observation.ne_representative = z_lm
                    #print('Map observation made at, Northings = ', new_observation.ne_representative[0], 'm, Eastings =', new_observation.ne_representative[1], 'm')  
                    print("object")
                    corner_training.append(new_observation)
                    #print(values)

                else:
                    error = fit_line_to_points(new_observation.data_filled, fit_error_tolerance_wall)

                    if error is not None:
                        new_observation.label = 'not corner'
                        new_observation.ne_representative = z_lm
                        #print('Map observation made at, Northings = ', new_observation.ne_representative[0], 'm, Eastings =', new_observation.ne_representative[1], 'm')  
                        print("---------------------------------print2---------------------------------")
                        corner_training.append(new_observation)
        else:
            new_observation.label = 'not corner'
            new_observation.ne_representative = z_lm
            #print('Map observation made at, Northings = ', new_observation.ne_representative[0], 'm, Eastings =', new_observation.ne_representative[1], 'm')  
            print("---------------------------------print2---------------------------------")
            corner_training.append(new_observation)


    return corner_training


def format_scan_object(filepath, threshold = 0.001, fit_error_tolerance = 0.01, fit_error_tolerance_wall = 0.005):
    fit_error = None
    variables = ImportLog(filepath)
    r = variables.extract_data("/lidar", ["message", "ranges"])
    theta = variables.extract_data("/lidar", ["message", "angles"])
    timestamps = variables.extract_data("/groundtruth", ["timestamp"])

    ###########________random bull shit which fixes formating_______###########

    if not isinstance(r, list):
        raise TypeError("Expected r to be a list, got {} instead.".format(type(r)))
    if not isinstance(theta, list):
        raise TypeError("Expected theta to be a list, got {} instead.".format(type(theta)))

    if len(r) == 0 or len(theta) == 0:
        raise ValueError("Extracted r or theta is empty. Ensure your logs have valid data.")

    if isinstance(r[0], list) or isinstance(r[0], np.ndarray):
        r[0] = np.array(r[0])
    else:
        r = [np.array(r)]  

    if isinstance(theta[0], list) or isinstance(theta[0], np.ndarray):
        theta[0] = np.array(theta[0])
    else:
        theta = [np.array(theta)]  

    observation = np.column_stack((r[0], theta[0]))  
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

        observation = np.column_stack((r[i], theta[i]))  
        z_lm = Vector(2)

        z_lm[0], z_lm[1], loc = find_corner(observation, threshold)
        #print(z_lm, loc)

        new_observation = GPC_input_output(observation, None)
        values = new_observation.data
        #print(values)

        #values = values.astype(float)
        #print("Values[0]:",values[:,0],"Values[1]:",values[:,1],)

        ####plotting funcitons####
        # fig,ax = plt.subplots()
        # show_scan(p, lidar, observation)
        # ax.scatter(m_y, m_x,s=0.01)
        # plt.show()

            ####if the number of nan's is more than X of the total size then pass#######
        if np.count_nonzero(~np.isnan(values[:,0])) > 0.1 * len(values[:,0]):

            if loc is not None:
                new_observation.label = 'not object'
                new_observation.ne_representative = z_lm
                #print('Map observation made at, Northings = ', new_observation.ne_representative[0], 'm, Eastings =', new_observation.ne_representative[1], 'm')  
                print("corner")

                corner_training.append(new_observation)

            else: 
                z_lm[0], z_lm[1], r_val, fit_error = fit_circle_to_points(new_observation, fit_error_tolerance)
                print("object fit error is here::::::--------",fit_error)

                if r_val is not None:
                    new_observation.label = 'object'
                    new_observation.ne_representative = z_lm
                    #print('Map observation made at, Northings = ', new_observation.ne_representative[0], 'm, Eastings =', new_observation.ne_representative[1], 'm')  
                    print("object")
                    corner_training.append(new_observation)
                    #print(values)

                else:
                    error = fit_line_to_points(new_observation.data_filled, fit_error_tolerance_wall)

                    if error is not None:
                        new_observation.label = 'not object'
                        new_observation.ne_representative = z_lm
                        #print('Map observation made at, Northings = ', new_observation.ne_representative[0], 'm, Eastings =', new_observation.ne_representative[1], 'm')  
                        print("---------------------------------print2---------------------------------")
                        corner_training.append(new_observation)
        else:
            new_observation.label = 'not object'
            new_observation.ne_representative = z_lm
            #print('Map observation made at, Northings = ', new_observation.ne_representative[0], 'm, Eastings =', new_observation.ne_representative[1], 'm')  
            print("---------------------------------print2---------------------------------")
            corner_training.append(new_observation)


    # for i in range(len(corner_training)):
    #     print('Entry:', i, ', Class', corner_training[i].label, ', Size', corner_training[i].data_filled[:, 0].size)
    #     print('Data type: Radius', corner_training[i].data_filled[:, 0])
    #     print('Data type:Theta', corner_training[i].data_filled[:, 1])
    
    return corner_training

def format_scan_wall(filepath, threshold = 0.001, fit_error_tolerance = 0.01, fit_error_tolerance_wall = 1):
    fit_error = None
    variables = ImportLog(filepath)
    r = variables.extract_data("/lidar", ["message", "ranges"])
    theta = variables.extract_data("/lidar", ["message", "angles"])
    timestamps = variables.extract_data("/groundtruth", ["timestamp"])

    ###########________random bull shit which fixes formating_______###########

    if not isinstance(r, list):
        raise TypeError("Expected r to be a list, got {} instead.".format(type(r)))
    if not isinstance(theta, list):
        raise TypeError("Expected theta to be a list, got {} instead.".format(type(theta)))

    if len(r) == 0 or len(theta) == 0:
        raise ValueError("Extracted r or theta is empty. Ensure your logs have valid data.")

    if isinstance(r[0], list) or isinstance(r[0], np.ndarray):
        r[0] = np.array(r[0])
    else:
        r = [np.array(r)]  

    if isinstance(theta[0], list) or isinstance(theta[0], np.ndarray):
        theta[0] = np.array(theta[0])
    else:
        theta = [np.array(theta)]  

    observation = np.column_stack((r[0], theta[0]))  
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

        observation = np.column_stack((r[i], theta[i]))  
        z_lm = Vector(2)

        z_lm[0], z_lm[1], loc = find_corner(observation, threshold)
        #print(z_lm, loc)

        new_observation = GPC_input_output(observation, None)
        values = new_observation.data
        #print(values)

        #values = values.astype(float)
        #print("Values[0]:",values[:,0],"Values[1]:",values[:,1],)

        ####plotting funcitons####
        # fig,ax = plt.subplots()
        # show_scan(p, lidar, observation)
        # ax.scatter(m_y, m_x,s=0.01)
        # plt.show()

            ####if the number of nan's is more than X of the total size then pass#######
        if fit_error_tolerance_wall == 1:
                        new_observation.label = 'wall'
                        new_observation.ne_representative = z_lm
                        #print('Map observation made at, Northings = ', new_observation.ne_representative[0], 'm, Eastings =', new_observation.ne_representative[1], 'm')  
                        print("---------------------------------print2---------------------------------")
                        corner_training.append(new_observation)
        else:
            new_observation.label = 'not wall'
            new_observation.ne_representative = z_lm
            #print('Map observation made at, Northings = ', new_observation.ne_representative[0], 'm, Eastings =', new_observation.ne_representative[1], 'm')  
            print("---------------------------------print2---------------------------------")
            corner_training.append(new_observation)


    # for i in range(len(corner_training)):
    #     print('Entry:', i, ', Class', corner_training[i].label, ', Size', corner_training[i].data_filled[:, 0].size)
    #     print('Data type: Radius', corner_training[i].data_filled[:, 0])
    #     print('Data type:Theta', corner_training[i].data_filled[:, 1])
    
    return corner_training

def clean_data(a):
    target_size = a[0].data_filled[:, 0].size
    X_train = []
    y_train = []

    # Data preparation
    for i in range(len(a)):
        if a[i].label is None:
            continue
            
        data = a[i].data_filled[:, 0]
        if data.size < target_size:
            padded = np.pad(data, (0, target_size - data.size), 
                        'constant', constant_values=np.nan)
        else:
            padded = data[:target_size]
        
        if np.isnan(padded).any():
            continue
            
        X_train.append(padded)
        y_train.append(a[i].label)

    X_train_clean = np.array(X_train)
    y_train_clean = np.array(y_train)

    return X_train_clean, y_train_clean


def find_thetas(scans, model_name=None, wl = 1, wr = 1):
    """
    Trains a GaussianProcessClassifier with model-specific hyperparameters.
    
    Parameters:
        scans: List of GPC_input_output objects with `.data_filled` and `.label`.
        model_name: A string to control kernel hyperparameter initialization.
    
    Returns:
        theta_1: Length scale
        theta_0: Signal variance (square root of constant)
        gpc: Trained classifier
        X_train_clean, y_train_clean: Cleaned training data
    """
    # --- Extract consistent input length ---
    target_size = max(s.data_filled[:, 0].size for s in scans)

    X_train, y_train = [], []

    for sample in scans:
        if sample.label is None:
            continue

        data = sample.data_filled[:, 0]
        if data.size < target_size:
            data = np.pad(data, (0, target_size - data.size), 'constant', constant_values=np.nan)
        else:
            data = data[:target_size]

        if not np.isnan(data).any():
            X_train.append(data)
            y_train.append(sample.label)

    X_train_clean = np.array(X_train)
    y_train_clean = np.array(y_train)

    if len(X_train_clean) == 0:
        raise ValueError("No valid training data available after filtering.")

    # --- Define kernel settings ---
    default_settings = {
        'constant': 1.0,
        'length': 1.0,
        'constant_bounds': (1e-3, 1e3),
        'length_bounds': (1e-3, 1e3),
        'restarts': 1,
        'seed': 42
    }

    # Modify kernel parameters based on model_name
    try:
        model_idx = int(model_name) if model_name is not None else 0
    except:
        model_idx = 0

    factor = 1.0 - model_idx * 0.001  # Decrement constant and length scale per model
    settings = {
        'constant': default_settings['constant'] * factor,
        'length': default_settings['length'] * (1 + model_idx * 0.05),
        'constant_bounds': default_settings['constant_bounds'],
        'length_bounds': default_settings['length_bounds'],
        'restarts': 3 + (model_idx % 3),  # vary restarts a bit
        'seed': 100 + model_idx  # ensure different seeds
    }

    # kernel = ConstantKernel(settings['constant'], settings['constant_bounds']) * \
    #          RBF(settings['length'], settings['length_bounds'])
    kernel =  wl* RBF(wr)

    # --- Fit GPC ---
    # gpc = GaussianProcessClassifier(
    #     kernel=kernel,
    #     optimizer='fmin_l_bfgs_b',
    #     n_restarts_optimizer=settings['restarts'],
    #     random_state=settings['seed'],
    #     copy_X_train=False
    # )

    # --- Fit GPC ---
    gpc = GaussianProcessClassifier(
        kernel=kernel,
        optimizer=None,
        random_state=settings['seed'],
        copy_X_train=False
    )

    gpc.fit(X_train_clean, y_train_clean)

    # --- Extract and report optimized hyperparameters ---
    opt_kernel = gpc.kernel_
    theta_1 = opt_kernel.k2.length_scale
    theta_0 = np.sqrt(opt_kernel.k1.constant_value)
    nll = -gpc.log_marginal_likelihood_value_

    print(f'\n=== Model {model_name or "default"} ===')
    print(f'Optimized kernel: {opt_kernel}')
    print(f'Theta: [theta_0: {theta_0:.3f}, theta_1: {theta_1:.3f}]')
    print(f'Negative Log-Likelihood (NLL): {nll:.3f}')

    return theta_1, theta_0, gpc, X_train_clean, y_train_clean


######scans making the gpc then the data you are validating against#######
def find_thetas_cross_validate(scans, X_train, y_train, wl = 1, wr = 1):
    

    theta_1,theta_0, gpc, X_train,y_train = find_thetas(scans,model_name='11', wl=wl , wr=wr)
    score, rep = cross_validate_acc_and_rep(gpc, X_train, y_train)   # Get a performance metric

    return theta_1, theta_0, gpc, X_train, y_train, score, rep  # Add score to returns



def cross_validate(original_gpc, X_train_clean, y_train_clean):
    # Clone the trained model and preserve the fitted kernel
    kernel = original_gpc.kernel_
    
    # Build a new GPC with fixed kernel and NO further optimization
    model = GaussianProcessClassifier(kernel=kernel, optimizer=None)
    
    # Cross-validation without fitting again (scikit-learn handles this safely)
    scores = cross_val_score(model, X_train_clean, y_train_clean, cv=5)
    
    print(f"Mean accuracy: {scores.mean():.4f}")
    print(f"Repeatability: {1 - scores.std()/scores.mean():.4f}")
    rep = 1 - scores.std()/scores.mean()

    return scores.mean()

def cross_validate_acc_and_rep(original_gpc, X_train_clean, y_train_clean):
    # Clone the trained model and preserve the fitted kernel
    kernel = original_gpc.kernel_
    
    # Build a new GPC with fixed kernel and NO further optimization
    model = GaussianProcessClassifier(kernel=kernel, optimizer=None)
    
    # Cross-validation without fitting again (scikit-learn handles this safely)
    scores = cross_val_score(model, X_train_clean, y_train_clean, cv=5)
    
    print(f"Mean accuracy: {scores.mean():.4f}")
    print(f"Repeatability: {1 - scores.std()/scores.mean():.4f}")
    rep = 1 - scores.std()/scores.mean()

    return scores.mean(), rep


def combine_scans(*scans):
    """
    Combines multiple lists of GPC_input_output objects into a single list.

    Args:
        scans: Variable number of scan lists (each output from format_scan).

    Returns:
        combined_scans: A single list containing all GPC_input_output entries.
    """
    combined_scans = []
    for scan in scans:
        combined_scans.extend(scan)

    return combined_scans

def verify_model_differences():
    models = {
        '0_R_H': gpc_0_R_H,
        '0_R': gpc_0_R,
        '0': gpc_0,
        'H': gpc_H
    }
    
    print("\n=== Final Model Differences ===")
    for name, model in models.items():
        print(f"{name} kernel: {model.kernel_}")
        print(f"{name} thetas: [{np.sqrt(model.kernel_.k1.constant_value):.3f}, {model.kernel_.k2.length_scale:.3f}]")
        print(f"{name} NLL: {-model.log_marginal_likelihood_value_:.3f}\n")


def gpc_example(corner_0_noise, gpc_0, threshold=0.5, scan=None):
    """
    Calculate average probabilities across all scans or a specific scan in corner_0_noise
    
    Parameters:
    - corner_0_noise: List of GPC_input_output objects
    - gpc_0: Trained Gaussian Process Classifier
    - threshold: Probability threshold for classification
    - scan: None to average all scans, or index for specific scan
    
    Returns:
    - Tuple of (average_label, average_probabilities) or specific scan results
    """
    
    if scan is not None:
        # Process specific scan if index is provided
        first_scan = corner_0_noise[scan]
        first_scan_data = first_scan.data_filled
        
        show_scan(p, lidar, first_scan_data)
        new_observation = GPC_input_output(first_scan_data, None)
        
        probs = gpc_0.predict_proba([new_observation.data_filled[:,0]])
        print('New observation is ',new_observation.label, ', with probabilities', gpc_0.predict_proba([new_observation.data_filled[:,0]]))
        max_prob = np.max(probs)
        
        if max_prob >= threshold:
            label = gpc_0.classes_[np.argmax(probs)]
            print(f'Single scan is {label}, with probabilities {probs}')
            return label, probs
        else:
            return "nothing homie"
    else:
        # Process all scans and average probabilities
        all_probs = []
        valid_scans = 0
        
        for scan_data in corner_0_noise:
            scan_data_clean = scan_data.data_filled[:,0]
            
            # Skip if data is invalid
            if len(scan_data_clean) == 0 or np.all(np.isnan(scan_data_clean)):
                continue
                
            probs = gpc_0.predict_proba([scan_data_clean])
            all_probs.append(probs)
            valid_scans += 1
        
        if valid_scans == 0:
            return "No valid scans found"
            
        # Calculate average probabilities
        avg_probs = np.mean(np.array(all_probs), axis=0)
        max_avg_prob = np.max(avg_probs)
        
        if max_avg_prob >= threshold:
            avg_label = gpc_0.classes_[np.argmax(avg_probs)]
            print(f'Average across {valid_scans} scans: {avg_label}, with avg probabilities {avg_probs}')
            return avg_label, avg_probs
        else:
            return "Average probability below threshold"


def gpc_example_old(corner_0_noise, gpc_0,threshold = 0.5, scan = 0):
    # Extract the first scan (which is at index 0)
    first_scan = corner_0_noise[scan]
    # Access its components:
    first_scan_data = first_scan.data_filled  # The (r,θ) points

    #print(first_scan_data)

    show_scan(p, lidar, first_scan_data)
    # log the scan for classification
    new_observation = GPC_input_output(first_scan_data, None)

    #!!!!!!!!!!!!!!!!! Use classifier to judge if it is a corner or not, can increase this to be more conservative!!!!!!!!!
    if np.max(gpc_0.predict_proba([new_observation.data_filled[:,0]]))>=threshold:
        new_observation.label = (gpc_0.classes_[np.argmax(gpc_0.predict_proba([new_observation.data_filled[:,0]]))])
        print('New observation is ',new_observation.label, ', with probabilities', gpc_0.predict_proba([new_observation.data_filled[:,0]]))

        return new_observation.label, gpc_0.predict_proba([new_observation.data_filled[:,0]])
    else:
        return "nothing homie"




#0.0005
##corner training### 
c_corner_0_noise = format_scan_corner("logs/corner_perfect_lidar.json", 0.001,0.1,1)
c_wall_0_noise = format_scan_corner("logs/wall_perfect_lidar.json", 10,0.1,1)
c_object_0_noise = format_scan_corner("logs/object_perfect_lidar.json", 10,0.1,1)

c_corner_low_noise = format_scan_corner("logs/corner_1_deg_5mm.json", 0.001,0.1,1)
c_corner_high_noise = format_scan_corner("logs/corner_3deg_15mm.json", 0.001,0.1,1)

#########extended corner data#######
c_ranged_far = format_scan_corner("logs/Corner_range_far.json", 0,0.1,1)
c_ranged_near = format_scan_corner("logs/corner_range_near.json", 0,0.1,1)
c_rotaion = format_scan_corner("logs/corner_0_test1_rotation.json", 0,0.1,1)
c_side_left = format_scan_corner("logs/side_corner_left.json", 0,0.1,1)
c_side_right = format_scan_corner("logs/side_corner_right_real.json", 0,0.1,1)

c_wall_low_noise = format_scan_corner("logs/wall_1_deg_5mm.json", 100,0.1,1)
c_wall_high_noise = format_scan_corner("logs/wall_3deg_15mm.json", 100,0.1,1)

c_object_low_noise = format_scan_corner("logs/object_1_deg_5mm.json", 100,50,1)
c_object_high_noise = format_scan_corner("logs/object_3deg_15mm.json", 100,50,1)


##object training###
o_corner_0_noise = format_scan_object("logs/corner_perfect_lidar.json", 10,0.00001,1)
o_wall_0_noise = format_scan_object("logs/wall_perfect_lidar.json", 10,0.00001,1)
o_object_0_noise = format_scan_object("logs/object_perfect_lidar.json", 10,50,1)

o_object_low_noise = format_scan_object("logs/object_1_deg_5mm.json", 10,50,1)
o_object_high_noise = format_scan_object("logs/object_3deg_15mm.json", 10,50,1)

########extended object data######
o_ranged_far = format_scan_object("logs/object_range_far.json", 10,50,1)
o_ranged_near = format_scan_object("logs/Range_object_near.json", 10,50,1)
o_rotaion = format_scan_object("logs/object_0_extended_rotation.json", 10,50,1)
o_side_left = format_scan_object("logs/object_side_left.json", 10,50,1)
o_side_right = format_scan_object("logs/object_side_right.json", 10,50,1)




##wall training###
w_corner_0_noise = format_scan_wall("logs/corner_perfect_lidar.json", 10,0.1,10)
w_wall_0_noise = format_scan_wall("logs/wall_perfect_lidar.json", 10,0,1)
w_object_0_noise = format_scan_wall("logs/object_perfect_lidar.json", 10,0.1,0)

w_wall_low_noise = format_scan_wall("logs/wall_1_deg_5mm.json", 10,0,10)
w_wall_high_noise = format_scan_wall("logs/wall_3deg_15mm.json", 10,0,10)

#extended data
w_ranged_far = format_scan_wall("logs/range_wall_far.json", 10,0,1)
w_ranged_near = format_scan_wall("logs/range_wall_near.json", 10,0,1)
w_rotaion = format_scan_wall("logs/Wall_0_extended_rotation.json", 10,0,1)
w_side_left = format_scan_wall("logs/side_wall_left.json", 10,0,1)
w_side_right = format_scan_wall("logs/side_wall_right.json", 10,0,1)




#-----extras----
# object_0_noise = format_scan_corner("logs/object_perfect_lidar.json", 10,0.1,1)
# object_low_noise = format_scan_corner("logs/object_1_deg_5mm.json", 10,0.1,1)
# object_high_noise = format_scan_corner("logs/object_3deg_15mm.json", 10,0.1,1)
# corner_low_noise = format_scan_corner("logs/corner_1_deg_5mm.json", 0.001,0.1,1)
# object_low_noise = format_scan_corner("logs/object_1_deg_5mm.json", 10,0.1,1)
# wall_low_noise = format_scan_corner("logs/wall_1_deg_5mm.json", 10,0.1,1)
# wall_0_noise = format_scan_corner("logs/wall_perfect_lidar.json", 10,0.1,1)
# wall_low_noise = format_scan_corner("logs/wall_1_deg_5mm.json", 10,0.1,1)
# wall_high_noise = format_scan_corner("logs/wall_3deg_15mm.json", 10,0.1,1)

print("-----------------------testcombine_scan----------------")






# object_0 = combine_scans(o_corner_0_noise,o_wall_0_noise,o_object_0_noise)
# o_corner_theta1_0, o_corner_theta2_0, o_gpc_0, o_DataX_0,o_DataY_0 = find_thetas(object_0,model_name='2')

# wall_0 = combine_scans(w_corner_0_noise,w_wall_0_noise,w_object_0_noise)
# w_corner_theta1_0, w_corner_theta2_0, w_gpc_0, w_DataX_0,w_DataY_0 = find_thetas(wall_0,model_name='3')


corner_0 = combine_scans(c_corner_0_noise,c_wall_0_noise,c_object_0_noise)
c_corner_theta1_0, c_corner_theta2_0, c_gpc_0, c_DataX_0,c_DataY_0 = find_thetas(corner_0,model_name='1')

corner_0 = combine_scans(c_corner_0_noise,c_wall_0_noise,c_object_0_noise)
c_corner_theta1_0, c_corner_theta2_0, c_gpc_0, c_DataX_0,c_DataY_0 = find_thetas(corner_0,model_name='1')

corner_0 = combine_scans(c_corner_0_noise,c_wall_0_noise,c_object_0_noise)
c_corner_theta1_0, c_corner_theta2_0, c_gpc_0, c_DataX_0,c_DataY_0 = find_thetas(corner_0,model_name='1')

c_low_noise_DataX, c_low_noise_DataY = clean_data(combine_scans(c_corner_low_noise,c_wall_low_noise,c_object_low_noise))
c_high_noise_DataX, c_high_noise_DataY = clean_data(combine_scans(c_corner_high_noise,c_wall_high_noise,c_object_high_noise))

cc_ranged_far = combine_scans(c_ranged_far,corner_0)
cc_ranged_near = combine_scans(c_ranged_near,corner_0)
cc_rotaion = combine_scans(c_rotaion,corner_0)
cc_side_left = combine_scans(c_side_left,corner_0)
cc_side_right = combine_scans(c_side_right,corner_0)

c_ranged_far_only_DataX_0, c_ranged_far_only_DataY_0  = clean_data(combine_scans(c_ranged_far,c_wall_0_noise,c_object_0_noise))
c_ranged_near_only_DataX_0,c_ranged_near_only_DataY_0  = clean_data(combine_scans(c_ranged_near,c_wall_0_noise,c_object_0_noise))
c_rotaion_only_DataX_0,c_rotaion_only_DataY_0   = clean_data(combine_scans(c_rotaion,c_wall_0_noise,c_object_0_noise))
c_side_left_only_DataX_0,c_side_left_only_DataY_0  = clean_data(combine_scans(c_side_left,c_wall_0_noise,c_object_0_noise))
c_side_right_only_DataX_0,c_side_right_only_DataY_0  = clean_data(combine_scans(c_side_right,c_wall_0_noise,c_object_0_noise))

##centre test data gpc found#####
__,__, c_ranged_far_gpc_0, c_ranged_far_DataX_0,c_ranged_far_DataY_0 = find_thetas(cc_ranged_far,model_name='4')
__,__, c_ranged_near_gpc_0, c_ranged_near_DataX_0,c_ranged_near_DataY_0 = find_thetas(cc_ranged_near,model_name='5')
__,__, c_rotaion_gpc_0, c_rotaion_DataX_0,c_rotaion_DataY_0 = find_thetas(cc_rotaion,model_name='6')
__,__, c_side_left_gpc_0, c_side_left_DataX_0,c_side_left_DataY_0 = find_thetas(cc_side_left,model_name='7')
__,__, c_side_right_gpc_0, c_side_right_DataX_0,c_side_right_DataY_0 = find_thetas(cc_side_right,model_name='8')


##all together
__,__, All_gpc_0, All_DataX_0,All_DataY_0 = find_thetas(combine_scans(c_corner_0_noise,c_wall_0_noise,c_object_0_noise,c_ranged_far,c_ranged_near,c_rotaion,c_side_left,c_side_right),model_name='9')
a,b, best_gpc_0, best_DataX_0,best_DataY_0 = find_thetas(combine_scans(c_corner_0_noise,c_wall_0_noise,c_object_0_noise,c_rotaion),model_name='10', wl= 5 , wr= 5)
c,d, best_gpc_0, best_DataX_0,best_DataY_0 = find_thetas(combine_scans(c_corner_0_noise,c_wall_0_noise,c_object_0_noise,c_rotaion),model_name='11', wl= 0.1 , wr= 1)


print("---------------------c_gpc_0----------------------")
print("c_gpc_0 vs c_DataX_0")
cross_validate(c_gpc_0, c_DataX_0,c_DataY_0)
print("c_gpc_0 vs c_ranged_far_only_DataX_0")
cross_validate(c_gpc_0, c_ranged_far_only_DataX_0, c_ranged_far_only_DataY_0)
print("c_gpc_0 vs c_ranged_near_only_DataX_0")
cross_validate(c_gpc_0, c_ranged_near_only_DataX_0,c_ranged_near_only_DataY_0)
print("c_gpc_0 vs c_rotaion_only_DataX_0")
cross_validate(c_gpc_0, c_rotaion_only_DataX_0,c_rotaion_only_DataY_0 )
print("c_gpc_0 vs c_side_left_only_DataX_0")
cross_validate(c_gpc_0, c_side_left_only_DataX_0,c_side_left_only_DataY_0)
print("c_gpc_0 vs c_side_right_only_DataX_0")
cross_validate(c_gpc_0, c_side_right_only_DataX_0,c_side_right_only_DataY_0)


print("----------------validate vs themselves---------------")
print("----------c_ranged_far_gpc_0 vs c_ranged_far_only_DataX_0")
cross_validate(c_ranged_far_gpc_0, c_ranged_far_only_DataX_0, c_ranged_far_only_DataY_0)
print("----------c_ranged_near_gpc_0 vs c_ranged_near_DataX_0")
cross_validate(c_ranged_near_gpc_0, c_ranged_near_DataX_0,c_ranged_near_DataY_0)
print("----------c_rotaion_gpc_0 vs c_rotaion_DataX_0")
cross_validate(c_rotaion_gpc_0, c_rotaion_DataX_0,c_rotaion_DataY_0)
print("----------c_side_left_gpc_0 vs c_side_left_DataX_0")
cross_validate(c_side_left_gpc_0, c_side_left_DataX_0,c_side_left_DataY_0)
print("----------c_side_right_gpc_0 vs c_side_right_DataX_0")
cross_validate(c_side_right_gpc_0, c_side_right_DataX_0,c_side_right_DataY_0)


print("---------------------ALL----------------------")
print("c_gpc_0 vs c_ranged_far_only_DataX_0")
cross_validate(All_gpc_0, c_ranged_far_only_DataX_0, c_ranged_far_only_DataY_0)
print("c_gpc_0 vs c_ranged_near_only_DataX_0")
cross_validate(All_gpc_0, c_ranged_near_only_DataX_0,c_ranged_near_only_DataY_0)
print("c_gpc_0 vs c_rotaion_only_DataX_0")
cross_validate(All_gpc_0, c_rotaion_only_DataX_0,c_rotaion_only_DataY_0 )
print("c_gpc_0 vs c_side_left_only_DataX_0")
cross_validate(All_gpc_0, c_side_left_only_DataX_0,c_side_left_only_DataY_0)
print("c_gpc_0 vs c_side_right_only_DataX_0")
cross_validate(All_gpc_0, c_side_right_only_DataX_0,c_side_right_only_DataY_0)
print("c_gpc_0 vs c_DataX_0")
cross_validate(All_gpc_0, c_DataX_0,c_DataY_0)


print("---------------------Best----------------------")
print("best_gpc_0 vs c_ranged_far_only_DataX_0")
cross_validate(c_rotaion_gpc_0, c_ranged_far_only_DataX_0, c_ranged_far_only_DataY_0)
print("best_gpc_0 vs c_ranged_near_only_DataX_0")
cross_validate(c_rotaion_gpc_0, c_ranged_near_only_DataX_0,c_ranged_near_only_DataY_0)
print("best_gpc_0 vs c_rotaion_only_DataX_0")
cross_validate(c_rotaion_gpc_0, c_rotaion_DataX_0,c_rotaion_DataY_0)
print("best_gpc_0 vs c_side_left_only_DataX_0")
cross_validate(c_rotaion_gpc_0, c_side_left_only_DataX_0,c_side_left_only_DataY_0)
print("best_gpc_0 vs c_side_right_only_DataX_0")
cross_validate(c_rotaion_gpc_0, c_side_right_only_DataX_0,c_side_right_only_DataY_0)
print("best_gpc_0 vs c_DataX_0")
cross_validate(c_rotaion_gpc_0, c_DataX_0,c_DataY_0)


# Define ranges for wl and wr
wl = 1.85  # 0.1 to 0.9 in steps of 0.1
wr_values = np.arange(0.1,3.0, 0.1)   # 1.0 to 4.5 in steps of 0.5

# Initialize a grid to store scores
scores_0 = np.zeros((len(wr_values)))
scores_low = np.zeros((len(wr_values)))
scores_high = np.zeros((len(wr_values)))
reps_0 = np.zeros((len(wr_values)))
reps_low = np.zeros((len(wr_values)))
reps_high = np.zeros((len(wr_values)))
wr_values_store = np.zeros((len(wr_values)))


#def find_thetas_cross_validate(scans, X_train, y_train, wl = 1, wr = 1):

# Iterate over all combinations
for j, wr in enumerate(wr_values):
    _, _, _, _,__, score_0, rep_0 = find_thetas_cross_validate(
        combine_scans(c_corner_0_noise, c_wall_0_noise, c_object_0_noise, c_rotaion),
        c_DataX_0,
        c_DataY_0,
        wl=wl,
        wr=wr
    )
    _, _, _, _,__, score_low, rep_low = find_thetas_cross_validate(
        combine_scans(c_corner_0_noise, c_wall_0_noise, c_object_0_noise, c_rotaion),
        c_low_noise_DataX,
        c_low_noise_DataY,
        wl=wl,
        wr=wr
    )
    _, _, _, _,__, score_high, rep_high = find_thetas_cross_validate(
        combine_scans(c_corner_0_noise, c_wall_0_noise, c_object_0_noise, c_rotaion),
        c_high_noise_DataX,
        c_high_noise_DataY,
        wl=wl,
        wr=wr
    )

    scores_0[j] = score_0
    reps_0[j] = rep_0
    scores_low[j] = score_low
    reps_low[j] = rep_low
    scores_high[j] = score_high
    reps_high[j] = rep_high    # Store the score

    wr_values_store[j] = wr


#print(wl_values_store)
print(wr_values_store)


max_idx_0 = np.argmax(scores_0)  # Index of max value in flattened array
max_idx_low = np.argmax(scores_low)
max_idx_high = np.argmax(scores_high)

max_0 = scores_0[max_idx_0]
max_low = scores_low[max_idx_low]
max_high = scores_high[max_idx_high]
print("max_0",max_0)
print("max_low",max_low)
print("max_high",max_high)




plt.axvline(x=wr_values_store[np.argmax(scores_0)], color='b', linestyle='-', alpha=0.3)
plt.axvline(x=wr_values_store[np.argmax(scores_low)], color='g', linestyle='--', alpha=0.3)
plt.axvline(x=wr_values_store[np.argmax(scores_high)], color='r', linestyle=':', alpha=0.3)

plt.figure(figsize=(10, 6))
plt.plot(wr_values_store, scores_0, 'b-', label='No Noise', linewidth=2)
plt.plot(wr_values_store, scores_low, 'g--', label='Low Noise', linewidth=2)
plt.plot(wr_values_store, scores_high, 'r:', label='High Noise', linewidth=2)

plt.xlabel('RBF Length Scale (wl)', fontsize=12)
plt.ylabel('Accuracy (Score)', fontsize=12)
plt.title('Impact of Length Scale (wl) on Accuracy at Different Noise Levels', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()


plt.figure(figsize=(10, 6))
plt.plot(wr_values_store, reps_0, 'b-', label='No Noise', linewidth=2)
plt.plot(wr_values_store, reps_low, 'g--', label='Low Noise', linewidth=2)
plt.plot(wr_values_store, reps_high, 'r:', label='High Noise', linewidth=2)

plt.xlabel('RBF Length Scale (wl)', fontsize=12)
plt.ylabel('Repeatability', fontsize=12)
plt.title('Impact of Length Scale (wl) on Repeatability at Different Noise Levels', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()



