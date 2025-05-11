import json
import numpy as np
import math
import matplotlib.pyplot as plt
from Libraries import *

# from src.Libraries.model_feeg6043 import RangeAngleKinematics, t2v, v2t, GPC_input_output
# from src.Libraries.plot_feeg6043 import plot_2dframe, show_observation
# from src.Libraries.math_feeg6043 import polar2cartesian, cartesian2polar, HomogeneousTransformation, l2m, Vector,Inverse, Matrix
import copy
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessClassifier #####from any python you learn#######
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report


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


def format_scan(filepath, label):
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
    corner_example = GPC_input_output(observation, label)
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
        new_observation = GPC_input_output(observation, label)
        corner_training.append(new_observation)
        
    
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

    return sklearn_theta_1,sklearn_theta_0, gpc_corner, X_train_clean, y_train_clean


def cross_validate(gpc_corner,X_train_clean,y_train_clean):

    ### i think gpc_corner is an instance of the kernal which we train and the thetas are auto-populated  usinging the data from GaussianProcessClassifier::
    print("Score",gpc_corner.score(X_train_clean, y_train_clean))
    print("classes",gpc_corner.classes_)

    ### Evaluate the model using cross-validation
    scores = cross_val_score(gpc_corner, X_train_clean, y_train_clean, cv=5)
    print("Cross-validated accuracy scores:", scores)
    print("Mean accuracy:", scores.mean())

    print("Standard deviation of accuracy:", scores.std())
    print("Mean accuracy (std):", scores.mean(), "+/-", scores.std())
    print("Repeatability (1 - std/mean):", 1 - scores.std()/scores.mean())

    
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

# #0.0005
# corner_0_noise = format_scan_corner("logs/corner_perfect_lidar.json", 0.001,0.1,1)
# corner_low_noise = format_scan_corner("logs/corner_1_deg_5mm.json", 0.001,0.1,1)
# corner_high_noise = format_scan_corner("logs/corner_3deg_15mm.json", 0.001,0.1,1)

# wall_0_noise = format_scan_corner("logs/wall_perfect_lidar.json", 1,0.1,1)
# wall_low_noise = format_scan_corner("logs/wall_1_deg_5mm.json", 1,0.1,1)
# wall_high_noise = format_scan_corner("logs/wall_3deg_15mm.json", 1,0.1,1)

# object_0_noise = format_scan_corner("logs/object_perfect_lidar.json", 1,0.1,1)
# object_low_noise = format_scan_corner("logs/object_1_deg_5mm.json", 1,0.1,1)
# object_high_noise = format_scan_corner("logs/object_3deg_15mm.json", 1,0.1,1)


# corner_0_R_H = combine_scans(corner_high_noise,corner_low_noise,corner_0_noise,wall_high_noise,wall_low_noise,wall_0_noise,object_high_noise,object_low_noise,object_0_noise)
# corner_theta1_0_R_H, corner_theta2_0_R_H, gpc_0_R_H,DataX_0_R_H_,DataY_0_R_H_ = find_thetas(corner_0_R_H)

# corner_0_R = combine_scans(corner_low_noise,corner_0_noise,wall_low_noise,wall_0_noise,object_low_noise,object_0_noise)
# corner_theta1_0_R, corner_theta2_0_R, gpc_0_R, __,__ = find_thetas(corner_0_R)

# corner_0 = combine_scans(corner_0_noise,wall_0_noise,object_0_noise)
# corner_theta1_0, corner_theta2_0, gpc_0, __,__ = find_thetas(corner_0)

# corner_H = combine_scans(corner_high_noise,wall_high_noise,object_high_noise)
# corner_theta1_H, corner_theta2_H, gpc_H, __,__ = find_thetas(corner_H)


# #print("object_theta1:",object_theta1, "object_theta2:",object_theta2)
# print("corner_theta1_0_R_H:",corner_theta1_0_R_H, "corner_theta2_0_R_H:",corner_theta2_0_R_H)
# print("corner_theta1_0_R:",corner_theta1_0_R, "corner_theta2_0_R:",corner_theta2_0_R)
# print("corner_theta1_0:",corner_theta1_0, "corner_theta2_0:",corner_theta2_0)
# print("corner_theta1_H:",corner_theta1_H, "corner_theta2_H:",corner_theta2_H)
# #print("wall_theta1:",wall_theta1, "wall_theta2:",wall_theta2)


# meanacc = cross_validate(gpc_0_R_H,DataX_0_R_H_,DataY_0_R_H_)