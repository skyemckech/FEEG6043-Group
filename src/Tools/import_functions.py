import json
import numpy as np
import math
import matplotlib.pyplot as plt
from datetime import datetime
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
    fig,ax = plt.subplots()
        
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
    plt.title("fuck this team")
    plt.show()

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

    target_size = 120  # or define manually
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

def add_noise(variance, mean, number):
    # Add random normal noise
    variance = max(0, variance)
    noise = np.random.normal(mean, np.sqrt(variance), number) 
    # first is the mean of the normal distribution you are choosing from
    # second is the standard deviation of the normal distribution
    # third is the number of elements you get in array noise
    return noise

def clean_data(lidar_data):
    for i in range(len(lidar_data)):
        data = lidar_data[i].data_filled[:, 0]
        if data.size < 120:
            # pad with NaNs or zeros
            padded = np.pad(data, (0, 120 - data.size), 'constant', constant_values=np.nan)
        else:
            # trim to target size
            padded = data[:120]
        return np.vstack(padded, lidar_data.label)

def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def check_landmarks(new_landmark, graph, acceptance_radius):
    distances = []
    old_landmarks = graph.landmark
    ref_x, ref_y = new_landmark
    for x, y in old_landmarks:
        distance = euclidean_distance(ref_x, ref_y, x, y)
        distances.append(distance)
    indices = [i for i, x in enumerate(distances) if x < acceptance_radius]
    if len(indices) != 0:
        final_id = graph.landmark_id_array[min(indices)]
        new = False
    elif len(graph.landmark_id_array) != 0:
        final_id = max(graph.landmark_id_array) + 1
        new = True
    else:
        final_id = 0
        new = True
    return new, final_id

def juice_graph(graph):
    graph_opt = graph
    initial_residual = 100 #just needs to be a big number to avoid triggering convergence if the first iteration has large residuals
    initial_flag = True

    residual_threshold = 1E-12 #if result changes by <1
    delta_threshold = 1/10 #if result changes by <1
    lim_iterations = 20

    n_iterations = 0
    delta_residual = initial_residual
    residual = initial_residual

    visualise_flag = False
    iteration_continue = True 
    residual_continue = True
    converge_continue = True

    cpu_start_solver = datetime.now()

    graph_opt = graphslam_backend(graph)

    while iteration_continue and residual_continue and converge_continue:    
        graph_opt.solve()
        
        prev_residual = residual
        residual = graph_opt.residual

        delta_residual = abs((prev_residual - residual) /prev_residual)
        n_iterations += 1

        print('**************  Residual = ',residual,' ***************')        
        residual_continue = (residual > residual_threshold)
        print('Residual above threshold?',residual_continue)    
        
        print('************** Iteration = ',n_iterations,' ***************')
        iteration_continue = (n_iterations <= lim_iterations)
        print('Iterations below limit?',iteration_continue)
        
        print('********* Delta Residual = ',delta_residual,' ***************')
        converge_continue = (delta_residual > delta_threshold)
        print('Residual still changing?',converge_continue)
        
        #reconstruct the graph with these nodes
        graph_opt = graphslam_frontend( graph_opt )   # Task
        graph_opt.construct_graph() # Task
        graph_opt = graphslam_backend( graph_opt )    # Task
        

    cpu_end_solver = datetime.now()
    delta =  cpu_end_solver - cpu_start_solver       
    print('********* Final solution took:',(delta.total_seconds()),'s ***************')      


    return graphslam_frontend( graph_opt )

def reduce_graph(graph):
    graph_opt = graph
    initial_residual = 100 #just needs to be a big number to avoid triggering convergence if the first iteration has large residuals
    initial_flag = True

    residual_threshold = 1E-12 #if result changes by <1
    delta_threshold = 1/1000 #if result changes by <1
    lim_iterations = 20

    n_iterations = 0
    delta_residual = initial_residual
    residual = initial_residual

    visualise_flag = False
    iteration_continue = True 
    residual_continue = True
    converge_continue = True

    cpu_start_solver = datetime.now()

    graph_opt = graphslam_backend(graph)
    pose_graph = graph_opt.reduce2pose(visualise_flag)
    l = graph_opt.n*3

    while iteration_continue and residual_continue and converge_continue:    
        graph_opt.solve()
        
        prev_residual = residual
        residual = graph_opt.residual

        delta_residual = abs((prev_residual - residual) /prev_residual)
        n_iterations += 1

        print('**************  Residual = ',residual,' ***************')        
        residual_continue = (residual > residual_threshold)
        print('Residual above threshold?',residual_continue)    
        
        print('************** Iteration = ',n_iterations,' ***************')
        iteration_continue = (n_iterations <= lim_iterations)
        print('Iterations below limit?',iteration_continue)
        
        print('********* Delta Residual = ',delta_residual,' ***************')
        converge_continue = (delta_residual > delta_threshold)
        print('Residual still changing?',converge_continue)
        
        #reconstruct the graph with these nodes
        graph_opt.state_vector[0:l] = pose_graph.state_vector # Task
        graph_opt.state_vector[l:] = Inverse(graph_opt.H[l:,l:])@(graph_opt.b[l:]+graph_opt.H[l:,0:l]@graph_opt.state_vector[0:l])
        graph_opt = graphslam_frontend( graph_opt )   # Task
        graph_opt.construct_graph(  ) # Task
        graph_opt = graphslam_backend( graph_opt )    # Task
        
        pose_graph = graph_opt.reduce2pose(visualise_flag)

    cpu_end_solver = datetime.now()
    delta =  cpu_end_solver - cpu_start_solver       
    print('********* Final solution took:',(delta.total_seconds()),'s ***************')      


    return graphslam_frontend( graph_opt )
            
def make_not_zero(value):
    if value == 0:
        value = 0.001
    return value

def plot_graph_square(graph, pose_ground_truth):
    m_e3 = Vector(2)
    m_e3[0] = 0
    m_e3[1] = 0

    m_e0 = Vector(2)
    m_e0[0] = 2
    m_e0[1] = 0

    m_e1 = Vector(2)
    m_e1[0] = 2
    m_e1[1] = 2

    m_e2 = Vector(2)
    m_e2[0] = 0
    m_e2[1] = 2
    map_ground_truth = [m_e0, m_e1, m_e2, m_e3]
    map_labels = ['m1', 'm2', 'm3', 'm4']

    em = Vector(2)
    H_em = HomogeneousTransformation(em,0)

    plot_graph(graph, pose_ground_truth, H_em, map_ground_truth, map_labels)
