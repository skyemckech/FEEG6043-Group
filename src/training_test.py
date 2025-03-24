import numpy as np
from Libraries.model_feeg6043 import RangeAngleKinematics, t2v, v2t
from Libraries.new_plot_feeg6043 import plot_2dframe, show_observation
from Libraries.math_feeg6043 import polar2cartesian, cartesian2polar, HomogeneousTransformation, l2m
import copy
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessClassifier #####from any python you learn#######
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

def lidar_scan(p_eb, environment_map, lidar, sigma_observe):
    """ Gets observations from the robot pose and the map.
    
    Parameters
    -----------
    p_eb = [3 x 1] Vector of floats
        The robot pose in the e frame as
            [[x
              y
              gamma]]
    environment_map = [l x 2] Matrix of floats
        A list observable element in the robots surroundings
        where each element is a transposed 2x1 vector of x, y coordinates in the environment frame
            [[x
              y]].T
    lidar = RangeAngleKinematics class
        A member of the RangeAngleKinematics that determines sensor properties
        range [min, max] in metres, and scan_fov in radians
        e.g., RangeAngleKinematics(x_bl, y_bl, distance_range = [0.1, 1], scan_fov = np.deg2rad(90)) 
        
    sigma_observe = [2 x 2] Matrix of floats
        Observation model for linear noise with range
        [[sigma_xr sigma_xtheta]
         [sigma_yr sigma_ytheta]]
        
    Returns
    -------
    observations: [n_beams x 2] Matrix of floats
        The observations as an array of (range, bearing) points. 
    observations_std: [n_beams x 1] Matrix of floats
        The observations set as an array of (range_std, bearing) points.         
    """
    m_range = [] # range and bearing to map elements
    m_bearing = []    
    
    z_range = [] # range and bearing measurements (resamples at resolution and with some noise)
    z_range_std = [] # range and bearing measurements (resamples at resolution and with some noise)
    z_bearing = []        
    
    for i in range(len(environment_map)):
        t_em = environment_map[[i],:].T
        z_lm, sigma_rtheta, t_lm, sigma_xy = lidar.loc_to_rangeangle(p_eb, t_em, sigma_observe)
        m_range.append(z_lm[0])
        m_bearing.append(z_lm[1])   
       
    #sampling the map
    bearing_resolution = np.linspace(-lidar.scan_fov/2,lidar.scan_fov/2,lidar.n_beams)

    #picks the nearest map entity and adds range based noise. Bearing noise is simulated to be half the beam width (i.e., (fov/n_beams)) 
    for theta in bearing_resolution:        
        if ~np.all(np.isnan(m_bearing)):
            i = np.nanargmin(abs((theta -m_bearing)))
        
            if abs(theta - m_bearing[i]) < lidar.scan_fov/(2*lidar.n_beams):
                z_range.append(m_range[i]+np.random.normal(0,m_range[i]*sigma_observe[0,0],1))
                z_range_std.append(abs(m_range[i]*sigma_observe[0,0]))
                z_bearing.append(theta)
            else:
                z_range.append(np.nan)
                z_range_std.append(np.nan)
                z_bearing.append(theta)
        else:
            z_range.append(np.nan)
            z_range_std.append(np.nan)
            z_bearing.append(theta)
   
    observations = l2m([z_range,z_bearing]) 
    observations_std = l2m([z_range_std, z_bearing]) 
    
    return observations, observations_std


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
            #show_observation(H_eb,t2v(lidar.H_bl.H@v2t(t_lm)),Matrix(2,2),None,ax, show_lines)        
        
    else:        
        cf=plot_2dframe(['pose','b','b'],[H_eb.H,H_eb.H],False,False)
        
    plt.xlabel('Eastings, m')
    plt.ylabel('Northings, m')
    plt.axis('equal')

def find_corner(corner, threshold = 0.01):
    # identify the reference coordinate as the inflection point
    
    # Step 1: Compute slope
    slope = np.gradient(corner.data[:, 0])
    
    # Step 2: Compute the second derivative (curvature)
    curvature = np.gradient(slope)
    
    # Step 3: Check if criteria is more than threshold    
    print('Max inflection value is ',np.nanmax(abs(np.gradient(np.gradient(curvature)))), ': Threshold ',threshold)
    if np.nanmax(abs(np.gradient(np.gradient(curvature)))) > threshold:
        # compute index of inflection point    
        largest_inflection_idx = np.nanargmax(abs(np.gradient(np.gradient(curvature))))  ####wheres the larges inflection point#### for finding corners
        
        r = corner.data[largest_inflection_idx, 0]  # Radial distance at the largest curvature
        theta = corner.data[largest_inflection_idx, 1]  # Angle at the largest curvature
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


