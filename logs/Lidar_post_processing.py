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


# def show_scan(p_eb, lidar, observations, show_lines = True):
#     """ Plots observations, field of view and robot pose
#     """

#     ######################## Calculate FOV    
#     range_max = lidar.distance_range[1]
#     range_min = lidar.distance_range[0]    
#     fov = lidar.scan_fov
        
#     r_ = []
#     theta_ = []    
    
#     # for field of view
#     theta = np.linspace(-fov / 2, fov / 2, 30)    
        
#     for i in theta:
#         r_.append(range_max)
#         theta_.append(i)        
#     for i in reversed(theta):
#         r_.append(range_min)
#         theta_.append(i)    
#     r_.append(range_max)
#     theta_.append(-fov/2)

#     fov = l2m([r_,theta_])
#     ######################## Plot the FOV        
    
#     t_lm = Vector(2) # lidar frame measurement placeholder    
#     t_em = Vector(2) # environment frame measurement
    
#     fov_x = []
#     fov_y = []

#     H_eb = HomogeneousTransformation(p_eb[0:2],p_eb[2])
        
#     for z_fov in fov:    
#         t_lm[0],t_lm[1] = polar2cartesian(z_fov[0],z_fov[1])      
#         t_em = t2v((H_eb.H@lidar.H_bl.H)@v2t(t_lm))
    
#         fov_x.append(t_em[0])
#         fov_y.append(t_em[1])  
        
#     if show_lines == True: plt.plot(fov_y, fov_x,'orange')

#     if len(observations) != 0:            
#         for z_lm in observations:    
#             t_lm[0],t_lm[1] = polar2cartesian(z_lm[0],z_lm[1])
#             show_observation(H_eb,t2v(lidar.H_bl.H@v2t(t_lm)),Matrix(2,2),None,ax, show_lines)        
        
#     else:        
#         cf=plot_2dframe(['pose','b','b'],[H_eb.H,H_eb.H],False,False)
        
#     plt.xlabel('Eastings, m')
#     plt.ylabel('Northings, m')
#     plt.axis('equal')

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


    # for i in range(len(corner_training)):
    #     print('Entry:', i, ', Class', corner_training[i].label, ', Size', corner_training[i].data_filled[:, 0].size)
    #     print('Data type: Radius', corner_training[i].data_filled[:, 0])
    #     print('Data type:Theta', corner_training[i].data_filled[:, 1])
    
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

def format_scan_wall(filepath, threshold = 0.001, fit_error_tolerance = 0.01, fit_error_tolerance_wall = 0.005):
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
                new_observation.label = 'not wall'
                new_observation.ne_representative = z_lm
                #print('Map observation made at, Northings = ', new_observation.ne_representative[0], 'm, Eastings =', new_observation.ne_representative[1], 'm')  
                print("corner")

                corner_training.append(new_observation)

            else: 
                z_lm[0], z_lm[1], r_val, fit_error = fit_circle_to_points(new_observation, fit_error_tolerance)
                print("object fit error is here::::::--------",fit_error)

                if r_val is not None:
                    new_observation.label = 'not wall'
                    new_observation.ne_representative = z_lm
                    #print('Map observation made at, Northings = ', new_observation.ne_representative[0], 'm, Eastings =', new_observation.ne_representative[1], 'm')  
                    print("object")
                    corner_training.append(new_observation)
                    #print(values)

                else:
                    error = fit_line_to_points(new_observation.data_filled, fit_error_tolerance_wall)

                    if error is not None:
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



def find_thetas(a):

    target_size = a[0].data_filled[:, 0].size  # or define manually
    X_train = []
    y_train = []

    #makes clean data for any size array: (clean data is the data without the first instance will all zeros in it)
    for i in range(len(a)):
        data = a[i].data_filled[:, 0]
        if a[i].label is not None:
            if data.size < target_size:
                # pad with NaNs or zeros
                padded = np.pad(data, (0, target_size - data.size), 'constant', constant_values=np.nan)
            else:
                # trim to target size
                padded = data[:target_size]
            X_train.append(padded)
            y_train.append(a[i].label)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    X_train_clean = []
    y_train_clean = []

    for i in range(len(y_train)):
        if y_train[i] is not None:
            X_train_clean.append(X_train[i])
            y_train_clean.append(y_train[i])

    X_train_clean = np.array(X_train_clean)
    y_train_clean = np.array(y_train_clean)

    # gpc_corner is the instnace of the classifier which we used with the weighting comands
    kernel = 1.0 * RBF(1.0)
    gpc_corner = GaussianProcessClassifier(kernel=kernel,random_state=0).fit(X_train_clean, y_train_clean)

    ### i think gpc_corner is an instance of the kernal which we train and the thetas are auto-populated  usinging the data from GaussianProcessClassifier::
    print("Score",gpc_corner.score(X_train_clean, y_train_clean))
    print("classes",gpc_corner.classes_)

    # Obtain optimized kernel parameters
    sklearn_theta_1 = gpc_corner.kernel_.k2.get_params()['length_scale']
    sklearn_theta_0 = np.sqrt(gpc_corner.kernel_.k1.get_params()['constant_value'])

    print(f'Optimized theta = [{sklearn_theta_0:.3f}, {sklearn_theta_1:.3f}], negative log likelihood = {-gpc_corner.log_marginal_likelihood_value_:.3f}')

    return sklearn_theta_1,sklearn_theta_0



###WORK PROGRESSED!!!!########
def combine_test_data(q,w,e,r):

    size_example = np.column_stack((corner_training[i].data_filled[:, 0], corner_training[i].data_filled[:, 1]))
    corner_example = GPC_input_output(q[0], None)
    corner_training = [corner_example]

    ####name of the values
    for i in range(len(corner_training)):
        print('Entry:', i, ', Class', corner_training[i].label, ', Size', corner_training[i].data_filled[:, 0].size)
        print('Data type: Radius', corner_training[i].data_filled[:, 0])
        print('Data type:Theta', corner_training[i].data_filled[:, 1])

        ##funciton which appends data to big ass variable:
    new_observation.label = 'corner'
    new_observation.ne_representative = z_lm
    print('Map observation made at, Northings = ', new_observation.ne_representative[0], 'm, Eastings =', new_observation.ne_representative[1], 'm')  
    print("corner")

    corner_training.append(new_observation)

    return 

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

#0.0005
corner_0_noise = format_scan_corner("logs/corner_perfect_lidar.json", 0.001,0.1,1)
corner_low_noise = format_scan_corner("logs/corner_1_deg_5mm.json", 0.001,0.1,1)
corner_high_noise = format_scan_corner("logs/corner_3deg_15mm.json", 0.001,0.1,1)
# corner_c = format_scan_corner("logs/RoundObject.json",0.001,0.1,1)
# corner_d = format_scan_corner("logs/StaticCorner.json", 0.001,0.01,1)
# corner_e = format_scan_corner("logs/StaticRoundObject.json", 0.001,0.1,1)
# corner_f = format_scan_corner("logs/StaticWall.json", 0.001,0.1,1)
# corner_g = format_scan_corner("logs/StraightLinePlusCorner.json",0.001,0.1,1)
# corner_h = format_scan_corner("logs/StraightLinePlusCorner2.json", 0.001,0.01,1)
#corner_i = format_scan_corner("logs/CORNERS_CLASSIFIER_20250325_135248_log.json", 0.01,0.01,1)

#object_a = format_scan_object("logs/MovingCircle.json", 15,1,1)
#object_b = format_scan_object("logs/MovingCircleFast.json", 15,1,1)
# object_c = format_scan_object("logs/RoundObject.json",15,1,1)
# object_d = format_scan_object("logs/StaticCorner.json", 15,1,1)
# object_e = format_scan_object("logs/StaticRoundObject.json", 15,1,1)
# object_f = format_scan_object("logs/StaticWall.json", 15,1,1)
# object_g = format_scan_object("logs/StraightLinePlusCorner.json",15,1,1)
# object_h = format_scan_object("logs/StraightLinePlusCorner2.json", 15,1,1)
#object_i = format_scan_object("logs/CORNERS_CLASSIFIER_20250325_135248_log.json", 0.01,0.01,1)

#wall_a = format_scan_wall("logs/MovingCircle.json", 15,0.0001,1)
#wall_b = format_scan_wall("logs/MovingCircleFast.json", 15,0.0001,1)
# wall_c = format_scan_wall("logs/RoundObject.json",15,0.0001,1)
# wall_d = format_scan_wall("logs/StaticCorner.json", 15,0.0001,1)
# wall_e = format_scan_wall("logs/StaticRoundObject.json", 15,0.0001,1)
# wall_f = format_scan_wall("logs/StaticWall.json", 15,0.0001,1)
# wall_g = format_scan_wall("logs/StraightLinePlusCorner.json",15,0.0001,1)
# wall_h = format_scan_wall("logs/StraightLinePlusCorner2.json", 15,0.0001,1)
#wall_i = format_scan_wall("logs/CORNERS_CLASSIFIER_20250325_135248_log.json", 0.01,0.01,1)





print("-----------------------testprint----------------")
# object_r = combine_scans(object_a,object_b,object_c,object_d,object_e,object_f,object_g,object_h)
# object_theta1, object_theta2 = find_thetas(object_r)

# print("object_r")
# for i in range(len(object_r)):
#       print('Entry:', i, ', Class', object_r[i].label)


# corner_r = combine_scans(corner_a,corner_b,corner_c,corner_d,corner_e,corner_f,corner_g,corner_h)
# corner_theta1, corner_theta2 = find_thetas(corner_r)

# print("corner_r")
# for i in range(len(corner_r)):
#       print('Entry:', i, ', Class', corner_r[i].label)

# wall_r = combine_scans(wall_a,wall_b,wall_c,wall_d,wall_e,wall_f,wall_g,wall_h)
# wall_theta1, wall_theta2 = find_thetas(wall_r)

# print("wall_r")
# for i in range(len(wall_r)):
#       print('Entry:', i, ', Class', wall_r[i].label)
    #####More printing bollox#######
print("corner_0_noise")    
for i in range(len(corner_0_noise)):
      print('Entry:', i, ', Class', corner_0_noise[i].label)
print("corner_low_noise")
for i in range(len(corner_low_noise)):
      print('Entry:', i, ', Class', corner_low_noise[i].label)
print("corner_high_noise")
for i in range(len(corner_high_noise)):
      print('Entry:', i, ', Class', corner_high_noise[i].label)

# print("corner_a")
# for i in range(len(corner_d)):
#       print('Entry:', i, ', Class', corner_d[i].label)
# print("corner_a")
# for i in range(len(corner_e)):
#       print('Entry:', i, ', Class', corner_e[i].label)
# print("corner_a")
# for i in range(len(corner_f)):
#       print('Entry:', i, ', Class', corner_f[i].label)
# print("corner_a")
# for i in range(len(corner_g)):
#       print('Entry:', i, ', Class', corner_g[i].label)
# print("corner_a")
# for i in range(len(corner_h)):
#       print('Entry:', i, ', Class', corner_h[i].label)

print("-----------------------testcombine_scan----------------")
corner_O_R_H = combine_scans(corner_high_noise,corner_low_noise,corner_0_noise)
corner_theta1, corner_theta2 = find_thetas(corner_O_R_H)


# corner_r = combine_scans(corner_b)#,corner_b,corner_c,corner_d,corner_e,corner_f,corner_g,corner_h)
# corner_theta1, corner_theta2 = find_thetas(corner_r)

print("corner_O_R_H")
for i in range(len(corner_O_R_H)):
      print('Entry:', i, ', Class', corner_O_R_H[i].label)

# wall_r = combine_scans(wall_b)#,wall_b,wall_c,wall_d,wall_e,wall_f,wall_g,wall_h)
# wall_theta1, wall_theta2 = find_thetas(wall_r)

# print("wall_r")
# for i in range(len(wall_r)):
#       print('Entry:', i, ', Class', wall_r[i].label)



# print("object_r")
# for i in range(len(object_r)):
#       print('Entry:', i, ', Class', object_r[i].label)

# print("corner_r")
# for i in range(len(corner_r)):
#       print('Entry:', i, ', Class', corner_r[i].label)

# print("wall_r")
# for i in range(len(wall_r)):
#       print('Entry:', i, ', Class', wall_r[i].label)

#print("object_theta1:",object_theta1, "object_theta2:",object_theta2)
print("corner_theta1:",corner_theta1, "corner_theta2:",corner_theta2)
#print("wall_theta1:",wall_theta1, "wall_theta2:",wall_theta2)