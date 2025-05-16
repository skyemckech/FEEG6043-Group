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
from matplotlib.widgets import Button
from pyton_skin import data_manager
from collections import Counter 




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

def analyze_scans():
    # Get all corner scans
    corners = data_manager.get_scans_by_label('corner')
    print(f"Found {sum(len(v) for v in corners.values())} corner scans")
    
    # Get all non-corner scans
    non_corners = data_manager.get_scans_by_label('not_corner')
    print(f"Found {sum(len(v) for v in non_corners.values())} non-corner scans")
    
    # Example: Plot the first corner scan from each file
    for file_key, scans in corners.items():
        if scans:
            scan = scans[0]
            plt.figure()
            plt.scatter(scan['observation'][:, 1], scan['observation'][:, 0], s=1)
            plt.title(f"Corner scan from {file_key}")
            plt.show()

if __name__ == "__main__":
    analyze_scans()


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
def format_scan_lablee(filepath, threshold=0.001, fit_error_tolerance=0.01, fit_error_tolerance_wall=0.005):
    class LabelSelector:
        def __init__(self):
            self.label = None
        
        def corner(self, event):
            self.label = 'corner'
            plt.close()
        
        def not_corner(self, event):
            self.label = 'not corner'
            plt.close()

    # Initialize label selector outside the loop
    label_selector = LabelSelector()

    # ... (keep all your original data loading and validation code) ...
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


        # ... (keep your existing data processing code) ...

        if j < 1000:
            j = j + 1
            fig, ax = plt.subplots()
            show_scan(p, lidar, observation)
            ax.scatter(m_y, m_x, s=0.01)
            plt.title(loc)

            # Create fresh buttons for each plot
            ax_corner = plt.axes([0.4, 0.05, 0.1, 0.075])
            ax_not_corner = plt.axes([0.55, 0.05, 0.1, 0.075])
            
            btn_corner = Button(ax_corner, 'Corner')
            btn_not_corner = Button(ax_not_corner, 'Not Corner')
            
            # Connect buttons to the label selector instance
            btn_corner.on_clicked(label_selector.corner)
            btn_not_corner.on_clicked(label_selector.not_corner)
            
            plt.show()
            
            # Assign the selected label
            new_observation.label = label_selector.label
            print(f"Scan {j} labeled as: {new_observation.label}")
            corner_training.append(new_observation)


    return corner_training
    
#"logs/all_static_corners_&_walls_20250325_135405_log.json"
def format_scan_lable(filepath, threshold=0.001, fit_error_tolerance=0.01, fit_error_tolerance_wall=0.005):

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

        if j < 1000:
            j = j + 1
            fig, ax = plt.subplots()
            show_scan(p, lidar, observation)
            ax.scatter(m_y, m_x, s=0.01)
            plt.title(loc)

            # Add buttons
            ax_corner = plt.axes([0.4, 0.05, 0.1, 0.075])
            ax_not_corner = plt.axes([0.55, 0.05, 0.1, 0.075])
            
            btn_corner = Button(ax_corner, 'Corner')
            btn_not_corner = Button(ax_not_corner, 'Not Corner')
            
            btn_corner.on_clicked(label_selector.corner)
            btn_not_corner.on_clicked(label_selector.not_corner)
            
            plt.show()
            
            # Wait for user input
            new_observation.label = label_selector.label
            print(f"Scan {j} labeled as: {new_observation.label}")
            
            corner_training.append(new_observation)
        else:
            print("Maximum scans reached")
            break

    return corner_training

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
        
        if j < 1000:
            j = j + 1
            # fig,ax = plt.subplots()
            # show_scan(p, lidar, observation)
            # ax.scatter(m_y, m_x,s=0.01)
            # plt.title(loc)
            # # plt.show()
            # print(j)
        else:
            print("yummy cummy")

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

    # Debug: Count initial classes
    initial_classes = Counter(scan.label for scan in a if scan.label is not None)
    print(f"Initial class counts: {initial_classes}")

    for i in range(len(a)):
        if a[i].label is None:
            continue
            
        data = a[i].data_filled[:, 0]
        if data.size < target_size:
            padded = np.pad(data, (0, target_size - data.size), 'constant', constant_values=0)  # Pad with 0, not NaN
        else:
            padded = data[:target_size]
        
        X_train.append(padded)
        y_train.append(a[i].label)

    # Debug: Count final classes
    final_classes = Counter(y_train)
    print(f"Final class counts: {final_classes}")

    return np.array(X_train), np.array(y_train)


def clean_data_old(a):
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
    kernel = wl * RBF(length_scale=wr, length_scale_bounds='fixed')  # Disable optimization
    gpc = GaussianProcessClassifier(kernel=kernel, optimizer=None)    # No post-training tuning

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

    unique_classes = np.unique(y_train_clean)
    if len(unique_classes) < 2:
        print("\n[ERROR] Only one class detected in y_train_clean!")
        print(f"Classes present: {unique_classes}")
        print(f"Class counts: {Counter(y_train_clean)}")
        print("First 10 labels:", y_train_clean[:10])
        raise ValueError("Training data must contain at least 2 classes.")
    
    gpc.fit(X_train_clean, y_train_clean)  # Proceed only if validation passes

    return theta_1, theta_0, gpc, X_train_clean, y_train_clean


######scans making the gpc then the data you are validating against#######
def find_thetas_cross_validate(scans, X_train, y_train, wl = 1, wr = 1):
    

    #theta_1,theta_0, gpc, X_train,y_train = find_thetas(scans,model_name='11', wl=wl , wr=wr)
    #added danae
    theta_1, theta_0, gpc, _, _ = find_thetas(scans, wl=wl, wr=wr)

    score, rep = cross_validate_acc_and_rep(gpc, X_train, y_train)   # Get a performance metric

    return theta_1, theta_0, gpc, X_train, y_train, score, rep  # Add score to returns

def find_thetas_old_1(a, model_name=None):
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

    # Force different optimization paths based on model name
    if model_name == '1':
        kernel = ConstantKernel(1.02, constant_value_bounds=(1e-3, 1e3)) * \
                RBF(length_scale=1.1, length_scale_bounds=(1e-3, 1e3))
        n_restarts = 1
        random_state = 1
    elif model_name == '2':
        kernel = ConstantKernel(1.019, constant_value_bounds=(1e-2, 1e2)) * \
        RBF(length_scale=1.2, length_scale_bounds=(1e-2, 1e2))
        n_restarts = 4
        random_state = 4
    elif model_name == '3':
        kernel = ConstantKernel(1.018, constant_value_bounds=(1e-1, 1e1)) * \
                RBF(length_scale=1.3, length_scale_bounds=(1e-1, 1e1))
        n_restarts = 3
        random_state = 3
    elif model_name == '4':
        kernel = ConstantKernel(1.017, constant_value_bounds=(1e-3, 1e3)) * \
                RBF(length_scale=1.4, length_scale_bounds=(1e-3, 1e3))
        n_restarts = 2
        random_state = 2
    elif model_name == '5':
        kernel = ConstantKernel(1.016, constant_value_bounds=(1e-2, 1e2)) * \
        RBF(length_scale=1.5, length_scale_bounds=(1e-2, 1e2))
        n_restarts = 4
        random_state = 4
    elif model_name == '6':
        kernel = ConstantKernel(1.015, constant_value_bounds=(1e-1, 1e1)) * \
                RBF(length_scale=1.6, length_scale_bounds=(1e-1, 1e1))
        n_restarts = 3
        random_state = 3
    elif model_name == '7':
        kernel = ConstantKernel(1.014, constant_value_bounds=(1e-3, 1e3)) * \
                RBF(length_scale=1.7, length_scale_bounds=(1e-3, 1e3))
        n_restarts = 2
        random_state = 2
    elif model_name == '8':
        kernel = ConstantKernel(1.013, constant_value_bounds=(1e-2, 1e2)) * \
                RBF(length_scale=1.8, length_scale_bounds=(1e-2, 1e2))
        n_restarts = 4
        random_state = 4
    elif model_name == '9':
        kernel = ConstantKernel(1.012, constant_value_bounds=(1e-1, 1e1)) * \
                RBF(length_scale=1.9, length_scale_bounds=(1e-1, 1e1))
        n_restarts = 3
        random_state = 3
    elif model_name == '10':
        kernel = ConstantKernel(1.011, constant_value_bounds=(1e-3, 1e3)) * \
                RBF(length_scale=2.0, length_scale_bounds=(1e-3, 1e3))
        n_restarts = 2
        random_state = 2
    elif model_name == '11':
        kernel = ConstantKernel(1.01, constant_value_bounds=(1e-2, 1e2)) * \
        RBF(length_scale=2.1, length_scale_bounds=(1e-2, 1e2))
        n_restarts = 4
        random_state = 4
    elif model_name == '12':
        kernel = ConstantKernel(1.009, constant_value_bounds=(1e-1, 1e1)) * \
                RBF(length_scale=2.2, length_scale_bounds=(1e-1, 1e1))
        n_restarts = 3
        random_state = 3
    elif model_name == '13':
        kernel = ConstantKernel(1.008, constant_value_bounds=(1e-3, 1e3)) * \
                RBF(length_scale=2.3, length_scale_bounds=(1e-3, 1e3))
        n_restarts = 2
        random_state = 2
    elif model_name == '14':
        kernel = ConstantKernel(1.007, constant_value_bounds=(1e-2, 1e2)) * \
        RBF(length_scale=2.4, length_scale_bounds=(1e-2, 1e2))
        n_restarts = 4
        random_state = 4
    elif model_name == '15':
        kernel = ConstantKernel(1.006, constant_value_bounds=(1e-1, 1e1)) * \
                RBF(length_scale=2.5, length_scale_bounds=(1e-1, 1e1))
        n_restarts = 3
        random_state = 3
    elif model_name == '16':
        kernel = ConstantKernel(1.005, constant_value_bounds=(1e-3, 1e3)) * \
                RBF(length_scale=2.6, length_scale_bounds=(1e-3, 1e3))
        n_restarts = 2
        random_state = 2
    elif model_name == '17':
        kernel = ConstantKernel(1.004, constant_value_bounds=(1e-2, 1e2)) * \
                RBF(length_scale=2.7, length_scale_bounds=(1e-2, 1e2))
        n_restarts = 4
        random_state = 4
    elif model_name == '18':
        kernel = ConstantKernel(1.003, constant_value_bounds=(1e-1, 1e1)) * \
                RBF(length_scale=2.8, length_scale_bounds=(1e-1, 1e1))
        n_restarts = 3
        random_state = 3
    elif model_name == '19':
        kernel = ConstantKernel(1.002, constant_value_bounds=(1e-3, 1e3)) * \
                RBF(length_scale=2.9, length_scale_bounds=(1e-3, 1e3))
        n_restarts = 2
        random_state = 2
    else:
        kernel = ConstantKernel(1.001) * RBF(length_scale=1.0)
        n_restarts = 1
        random_state = 1

    # Lock the kernel parameters initially
    #kernel = kernel.clone_with_theta(kernel.theta)
    
    gpc = GaussianProcessClassifier(
        kernel=kernel,
        optimizer='fmin_l_bfgs_b',
        n_restarts_optimizer=n_restarts,
        random_state=random_state,
        copy_X_train=False  # Prevent data copying that might affect optimization
    ).fit(X_train_clean, y_train_clean)

    # Get optimized parameters
    optimized_kernel = gpc.kernel_
    theta_1 = optimized_kernel.k2.length_scale
    theta_0 = np.sqrt(optimized_kernel.k1.constant_value)

    print(f'\n=== {model_name} ===')
    print(f'Initial kernel: {kernel}')
    print(f'Optimized kernel: {optimized_kernel}')
    print(f'Theta: [{theta_0:.3f}, {theta_1:.3f}]')
    print(f'NLL: {-gpc.log_marginal_likelihood_value_:.3f}')

    return theta_1, theta_0, gpc, X_train_clean, y_train_clean

def find_thetas_old_2(a):

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

    # gpc_corner is the instnace of the classifier which we used with the weighting comands ###########Change These#############
    kernel = 1.0 * RBF(1.0)
    gpc_corner = GaussianProcessClassifier(kernel=kernel,random_state=0).fit(X_train_clean, y_train_clean)

    ### i think gpc_corner is an instance of the kernal which we train and the thetas are auto-populated  usinging the data from GaussianProcessClassifier::
    print("Score",gpc_corner.score(X_train_clean, y_train_clean))
    print("classes",gpc_corner.classes_)

    # Obtain optimized kernel parameters
    sklearn_theta_1 = gpc_corner.kernel_.k2.get_params()['length_scale']
    sklearn_theta_0 = np.sqrt(gpc_corner.kernel_.k1.get_params()['constant_value'])

    print(f'Optimized theta = [{sklearn_theta_0:.3f}, {sklearn_theta_1:.3f}], negative log likelihood = {-gpc_corner.log_marginal_likelihood_value_:.3f}')

    return sklearn_theta_1,sklearn_theta_0, gpc_corner, X_train_clean, y_train_clean


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

def cross_validate_old(original_gpc, X_train_clean, y_train_clean):
    # Create a true independent copy
    model = clone(original_gpc)
    model.set_params(**original_gpc.get_params())  # Preserve all parameters
    
    # Verify kernel preservation
    # print(f"\nOriginal kernel: {original_gpc.kernel}")
    # print(f"Cloned kernel: {model.kernel}")
    
    # Fit with verbose output
    model.fit(X_train_clean, y_train_clean)
    #print(f"Optimized kernel: {model.kernel_}")
    
    # Cross-validation
    scores = cross_val_score(model, X_train_clean, y_train_clean, cv=5)
    print(f"Mean accuracy: {scores.mean():.4f}")
    print(f"Repeatability: {1 - scores.std()/scores.mean():.4f}")
    
    return scores.mean()

def cross_validate_danae(gpc_corner,X_train_clean,y_train_clean):

    ### i think gpc_corner is an instance of the kernal which we train and the thetas are auto-populated  usinging the data from GaussianProcessClassifier::
    #print("Score",gpc_corner.score(X_train_clean, y_train_clean))
    #print("classes",gpc_corner.classes_)

    ### Evaluate the model using cross-validation
    gpc_corner.fit(X_train_clean, y_train_clean)
    scores = cross_val_score(gpc_corner, X_train_clean, y_train_clean, cv=5)
    #print("Cross-validated accuracy scores:", scores)
    print("Mean accuracy:", scores.mean())

    #print("Standard deviation of accuracy:", scores.std())
    #print("Mean accuracy (std):", scores.mean(), "+/-", scores.std())
    print("Repeatability (1 - std/mean):", 1 - scores.std()/scores.mean()) ## basicly saying where less of the data is likely to be!!!!

    
    # repeatability = evaluate_repeatability(scores)
    # print("Repeatability (1 - std/mean):", repeatability)

    # Generate a classification report based on the trained model
    print("Classification Report:")
    y_pred = gpc_corner.predict(X_train_clean)  # Predictions using the trained model
    print(classification_report(y_train_clean, y_pred))  # Detailed classification report

    return scores.mean()

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

# print("c_full_test_0")
# c_full_test_0 = format_scan_lablee("logs/full_test_0_noise_rr.json", 10,50,1)

# for i in range(len(c_full_test_0)):
#     print('Entry:', i, ', Class', c_full_test_0[i].label, ', Size', c_full_test_0[i].data_filled[:, 0].size)
#     # print('Data type: Radius', c_corner_0_noise[i].data_filled[:, 0])
#     # print('Data type:Theta', c_corner_0_noise[i].data_filled[:, 1])

# print("c_full_test_1")
# c_full_test_1 = format_scan_lablee("logs/full_test_1_noise_rr.json", 10,50,1)
# print("c_full_test_3")
# c_full_test_3 = format_scan_lablee("logs/full_test_3_noise_rr.json", 10,50,1)
# print("c_full_test_10")
# c_full_test_10 = format_scan_lablee("logs/full_test_10_noise_rr.json", 10,50,1)
print("c_full_test_10")
realsquarx3 = format_scan_lablee("logs/realsquare3x.json", 10,50,1)

#0.0005
##corner training### 
c_corner_0_noise = format_scan_corner("logs/corner_perfect_lidar.json", 0.001,0.1,1)
c_wall_0_noise = format_scan_corner("logs/wall_perfect_lidar.json", 10,0.1,1)
c_object_0_noise = format_scan_corner("logs/object_perfect_lidar.json", 10,0.1,1)

c_corner_low_noise = format_scan_corner("logs/corner_1_deg_5mm.json", 0.001,0.1,1)
c_corner_high_noise = format_scan_corner("logs/corner_3deg_15mm.json", 0.001,0.1,1)


# for i in range(len(c_corner_0_noise)):
#     print('Entry:', i, ', Class', c_corner_0_noise[i].label, ', Size', c_corner_0_noise[i].data_filled[:, 0].size)
#     print('Data type: Radius', c_corner_0_noise[i].data_filled[:, 0])
#     print('Data type:Theta', c_corner_0_noise[i].data_filled[:, 1])

########extended corner data#######
c_ranged_far = format_scan_corner("logs/Corner_range_far.json", 0,0.1,1)
c_ranged_near = format_scan_corner("logs/corner_range_near.json", 0,0.1,1)
c_rotaion = format_scan_corner("logs/corner_0_test1_rotation.json", 0,0.1,1)
c_side_left = format_scan_corner("logs/side_corner_left.json", 0,0.1,1)
c_side_right = format_scan_corner("logs/side_corner_right_real.json", 0,0.1,1)

c_wall_low_noise = format_scan_corner("logs/wall_1_deg_5mm.json", 100,0.1,1)
c_wall_high_noise = format_scan_corner("logs/wall_3deg_15mm.json", 100,0.1,1)

c_object_low_noise = format_scan_corner("logs/object_1_deg_5mm.json", 100,50,1)
c_object_high_noise = format_scan_corner("logs/object_3deg_15mm.json", 100,50,1)

c_object_vhigh_noise = format_scan_corner("logs/circle_x10R.json", 100,50,1)
c_wall_vhigh_noise = format_scan_corner("logs/wall_x10R.json", 100,0.1,1)
c_corner_vhigh_noise = format_scan_corner("logs/corner_x10R.json", 0.00001,0.1,1)



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


# realsquarx3 = combine_scans(c_corner_vhigh_noise,c_wall_vhigh_noise,c_object_vhigh_noise)



# object_0 = combine_scans(o_corner_0_noise,o_wall_0_noise,o_object_0_noise)
# o_corner_theta1_0, o_corner_theta2_0, o_gpc_0, o_DataX_0,o_DataY_0 = find_thetas(object_0,model_name='2')

# wall_0 = combine_scans(w_corner_0_noise,w_wall_0_noise,w_object_0_noise)
# w_corner_theta1_0, w_corner_theta2_0, w_gpc_0, w_DataX_0,w_DataY_0 = find_thetas(wall_0,model_name='3')


corner_0 = combine_scans(c_corner_0_noise,c_wall_0_noise,c_object_0_noise)
c_corner_theta1_0, c_corner_theta2_0, c_gpc_0, c_DataX_0,c_DataY_0 = find_thetas(corner_0,model_name='1')

c_low_noise_DataX, c_low_noise_DataY = clean_data(combine_scans(c_corner_low_noise,c_wall_low_noise,c_object_low_noise))
c_high_noise_DataX, c_high_noise_DataY = clean_data(combine_scans(c_corner_high_noise,c_wall_high_noise,c_object_high_noise))
c_10_noise_DataX, c_10_noise_DataY = clean_data(realsquarx3)

                                                                                                                                        # c_0_noise_DataX, c_0_noise_DataY = clean_data(realsquarx3)
# c_1_noise_DataX, c_1_noise_DataY = clean_data(c_full_test_1)
# c_3_noise_DataX, c_3_noise_DataY = clean_data(c_full_test_3)
# c_10_noise_DataX, c_10_noise_DataY = clean_data(c_full_test_10)

#full test
c_vhigh_noise_DataX, c_vhigh_noise_DataY = clean_data(combine_scans(c_corner_vhigh_noise,c_wall_vhigh_noise,c_object_vhigh_noise))

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

# print("----------------validate vs themselves---------------")
# print("----------c_ranged_far_gpc_0 vs c_DataX_0")
# cross_validate(c_ranged_far_gpc_0, c_DataX_0,c_DataY_0)
# print("----------c_ranged_near_gpc_0 vs c_DataX_0")
# cross_validate(c_ranged_near_gpc_0, c_DataX_0,c_DataY_0)
# print("----------c_rotaion_gpc_0 vs c_DataX_0")
# cross_validate(c_rotaion_gpc_0, c_DataX_0,c_DataY_0)
# print("----------c_side_left_gpc_0 vs c_DataX_0")
# cross_validate(c_side_left_gpc_0, c_DataX_0,c_DataY_0)
# print("----------c_side_right_gpc_0 vs c_DataX_0")
# cross_validate(c_side_right_gpc_0, c_DataX_0,c_DataY_0)

# c_0_noise_DataX, c_0_noise_DataY = clean_data(c_full_test_0)
# c_1_noise_DataX, c_1_noise_DataY = clean_data(c_full_test_1)
# c_3_noise_DataX, c_3_noise_DataY = clean_data(c_full_test_3)

# Define ranges for wl and wr
wl_values = np.arange(0.1, 5.0, 0.25)  # 0.1 to 0.9 in steps of 0.1
wr_values = np.arange(0.1, 5.0, 0.25)   # 1.0 to 4.5 in steps of 0.5

# Initialize a grid to store scores
scores_normal = np.zeros((len(wl_values), len(wr_values)))
scores_real_only = np.zeros((len(wl_values), len(wr_values)))
scores_real_rotation = np.zeros((len(wl_values), len(wr_values)))
scores_all = np.zeros((len(wl_values), len(wr_values)))
wl_values_store = np.zeros((len(wl_values)))
wr_values_store = np.zeros((len(wr_values)))


#def find_thetas_cross_validate(scans, X_train, y_train, wl = 1, wr = 1):

for i in range(len(realsquarx3)):
    print('Entry:', i, ', Class', realsquarx3[i].label, ', Size', realsquarx3[i].data_filled[:, 0].size)


#################Iterative bull shit################
c_corner_0_noise, c_wall_0_noise, c_object_0_noise, c_rotaion, c_10_noise_DataX, c_10_noise_DataY
# Iterate over all combinations
for i, wl in enumerate(wl_values):
    for j, wr in enumerate(wr_values):
        
        _, _, _, _,__, score1,__ = find_thetas_cross_validate(
                combine_scans(c_corner_0_noise, c_wall_0_noise, c_object_0_noise, c_rotaion),
                c_10_noise_DataX,
                c_10_noise_DataY,
                wl=wl,
                wr=wr
            )
        scores_normal[i, j] = score1  # Store the score
        wl_values_store[i] = wl
        wr_values_store[j] = wr

        _, _, _, _,__, score2,__ = find_thetas_cross_validate(
                combine_scans(realsquarx3, c_wall_0_noise, c_object_0_noise),
                c_10_noise_DataX,
                c_10_noise_DataY,
                wl=wl,
                wr=wr
            )
        scores_real_only[i, j] = score2  # Store the score


        _, _, _, _,__, score3,__ = find_thetas_cross_validate(
                combine_scans(realsquarx3,c_rotaion),
                c_10_noise_DataX,
                c_10_noise_DataY,
                wl=wl,
                wr=wr
            )

        scores_real_rotation[i, j] = score3  # Store the score



        _, _, _, _,__, score4,__ = find_thetas_cross_validate(
                combine_scans(c_corner_0_noise, c_wall_0_noise, c_object_0_noise, c_rotaion,realsquarx3),
                c_10_noise_DataX,
                c_10_noise_DataY,
                wl=wl,
                wr=wr
            )

        scores_all[i, j] = score4  # Store the score

print("wl_values_store")
print(wl_values_store)
print("wr_values_store")
print(wr_values_store)
print("scores_normal")
print(scores_normal)
print("scores_real_only")
print(scores_real_only)
print("scores_real_rotation")
print(scores_real_rotation)
print("scores_all")
print(scores_all)


# Create a grid for plotting
Wr, Wl = np.meshgrid(wl_values, wr_values)

max_idx_normal = np.argmax(scores_normal)  # Index of max value in flattened array
max_idx_real = np.argmax(scores_real_only)  # Index of max value in flattened array
max_idx_real_rotaion = np.argmax(scores_real_rotation)  # Index of max value in flattened array
max_idx_all = np.argmax(scores_all)  # Index of max value in flattened array

wl_idx_normal, wr_idx_normal = np.unravel_index(max_idx_normal, scores_normal.shape)  # Convert to 2D indices
wl_idx_real, wr_idx_real = np.unravel_index(max_idx_real, scores_real_only.shape)  # Convert to 2D indices
wl_idx_real_rotaion, wr_idx_real_rotaion = np.unravel_index(max_idx_real_rotaion, scores_real_rotation.shape)  # Convert to 2D indices
wl_idx_all, wr_idx_all = np.unravel_index(max_idx_all, scores_all.shape)  # Convert to 2D indices




best_wl_normal = wl_values[wl_idx_normal]
best_wr_normal = wr_values[wr_idx_normal]

best_wl_real = wl_values[wl_idx_real]
best_wr_real = wr_values[wr_idx_real]

best_wl_rotaion = wl_values[wl_idx_real_rotaion]
best_wr_rotaion = wr_values[wr_idx_real_rotaion]

best_wl_all = wl_values[wl_idx_all]
best_wr_all = wr_values[wr_idx_all]

best_score_normal = scores_normal[best_wl_normal, best_wr_normal]
best_score_real_only = scores_real_only[best_wl_real, best_wr_real]
best_score_rotaion = scores_real_rotation[best_wl_rotaion, best_wr_rotaion]
best_score_all = scores_all[best_wl_all, best_wr_all]

print("--------Normal Best Values----------")
print(f"Optimal weights: wl = {best_wl_normal:.2f}, wr = {best_wr_normal:.2f}")
print(f"Best score: {best_score_normal:.4f}")

print("--------A Best Values----------")
print(f"Optimal weights: wl = {best_wl_real:.2f}, wr = {best_wr_real:.2f}")
print(f"Best score: {best_score_real_only:.4f}")

print(f"Optimal weights: wl = {best_wl_rotaion:.2f}, wr = {best_wr_rotaion:.2f}")
print(f"Best score: {best_score_rotaion:.4f}")

print(f"Optimal weights: wl = {best_wl_all:.2f}, wr = {best_wr_all:.2f}")
print(f"Best score: {best_score_all:.4f}")

# Create meshgrid for 3D plotting



#################plotting bull shti ################
# # Set up the figure
# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111, projection='3d')

# # Plot the surface (grey)
# surf = ax.plot_surface(
#     Wl, Wr, scores.T,  # Transpose scores to match meshgrid
#     cmap='RdBu',      # Grey colormap
#     alpha=0.7,         # Slightly transparent
#     edgecolor='none'
# )

# # Highlight the optimal point (red)
# ax.scatter(
#     best_wl, best_wr, best_score,
#     color='red',
#     s=100,             # Marker size
#     label=f'Optimal: wl={best_wl:.2f}, wr={best_wr:.2f}\nScore={best_score:.3f}'
# )

# # Customize the plot
# ax.set_xlabel('wl (Kernel Multiplier)', fontsize=12)
# ax.set_ylabel('wr (RBF Length Scale)', fontsize=12)
# ax.set_zlabel('Score (e.g., Accuracy)', fontsize=12)
# ax.set_title('3D Surface Plot of Scores with Optimal Weights', fontsize=14)
# ax.legend(loc='upper right')

# # Add a colorbar for the surface
# fig.colorbar(surf, shrink=0.5, aspect=10, label='Score')

# plt.tight_layout()
# plt.show()




