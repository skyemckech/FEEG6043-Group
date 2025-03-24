import numpy as np
from Libraries.new_model_feeg6043 import RangeAngleKinematics, t2v, v2t
from Libraries.new_plot_feeg6043 import plot_2dframe, show_observation
from Libraries.new_math_feeg6043 import polar2cartesian, cartesian2polar, HomogeneousTransformation, l2m, Vector,Inverse, Matrix
import copy
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessClassifier #####from any python you learn#######
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
import matplotlib.pyplot as plt

pos_noise_std = 0.1
heading_noise_std = 10

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
            show_observation(H_eb,t2v(lidar.H_bl.H@v2t(t_lm)),Matrix(2,2),None,ax, show_lines)        
        
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


######## Simulate environment######################

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

# Create a scatter plot
plt.figure()
plt.scatter(m_x, m_y,s=0.1)
plt.axis('equal')
plt.title('Environment')
plt.show() 


######################## Observation model ##################
# locate lidar on robot (keep it simple)
x_bl = 0; y_bl = 0
t_bl = Vector(2)
t_bl[0] = x_bl
t_bl[1] = y_bl
H_bl = HomogeneousTransformation(t_bl,0)

# use Class from A1.3
lidar = RangeAngleKinematics(x_bl, y_bl, distance_range = [0.1, 1], scan_fov = np.deg2rad(60), n_beams = 30)
#################################################################



######################## Set the robot pose b in the e frame ################## all preset within webots code from blaire
p = Vector(3); 

p[0] = -0.7 #Northings
p[1] = -0.1 #Eastings
p[2] = np.deg2rad(270) #Heading (rad)
print('3x1 state vector:\n',p,'\n')
H_eb = HomogeneousTransformation(p[0:2],p[2])

#########observation model linear noise with range###############
sigma_observe = Matrix(2,2)
sigma_observe[0,0] = 0.01 #1% of range
sigma_observe[0,1] = 0
sigma_observe[1,0] = np.deg2rad(0.1) #0.1 degree per metre range      range and angle are uncerain within matrix
sigma_observe[1,1] = 0
print('2x2 measurement noise model:\n',sigma_observe,'\n')
#############################################################################

# compute observations with noise --------in polar for-------------[Radius, angle]-------
observation, _ = lidar_scan(p, environment_map, lidar, sigma_observe)
print(observation)

# store the data as an example of a 'corner' (label) ----------HERE LIDAR OBSEVATION!!!!!-----------
corner_scan = GPC_input_output(observation, 'corner')


###--------------Z_lm is the corner------------#########
z_lm = Vector(2)
z_lm[0], z_lm[1], loc = find_corner(corner_scan)

# update the landmark locations 
corner_scan.ne_representative = lidar.rangeangle_to_loc(p,z_lm)
print('Map observation made at, Northings = ',corner_scan.ne_representative[0],'m, Eastings =',corner_scan.ne_representative[1],'m')

# Show the landmark in the e-frame
H_el = HomogeneousTransformation()
H_el.H = H_eb.H@lidar.H_bl.H


# create a containor to store the GPC training data
corner_example = GPC_input_output(observation, None)
corner_training = [corner_example]


for i in range(40):    
    # determine basic pose for each corner
    if i<=10: # southwest corner
        p[0] = -0.5
        p[1] = -0.5
        p[2] = np.deg2rad(225)
    elif i<=20: # northwest corner
        p[0] = 0.5
        p[1] = -0.5
        p[2] = np.deg2rad(315)
    elif i<=30: # northwest corner
        p[0] = 0.5
        p[1] = 0.5
        p[2] = np.deg2rad(45)
    else:
        p[0] = -0.5
        p[1] = 0.5
        p[2] = np.deg2rad(135)

    # add random offsets
    p[0] += np.random.normal(-pos_noise_std, pos_noise_std)
    p[1] += np.random.normal(-pos_noise_std, pos_noise_std)
    p[2] += np.deg2rad(np.random.normal(-heading_noise_std, heading_noise_std))
    
    # compute observations with noise
    observation, _ = lidar_scan(p, environment_map, lidar, sigma_observe)

    ##Plotting the corner situation:
    fig,ax = plt.subplots()
    show_scan(p, lidar, observation)
    ax.scatter(m_y, m_x,s=0.01)
    #plt.show()

    # check if it is a corner with the inflection point
    new_observation = GPC_input_output(observation, None)
    
    threshold = 0.005 # can reduce to make less conservative
    z_lm[0], z_lm[1], loc = find_corner(new_observation, threshold)
    
    # if the bepoke model says returns a location, add to training data
    if loc is not None:
        # label corner and add to corner training set
        new_observation.label='corner'
        new_observation.ne_representative=z_lm
        print('Map observation made at, Northings = ',new_observation.ne_representative[0],'m, Eastings =',new_observation.ne_representative[1],'m')        
        corner_training.append(new_observation)
        
        # show pose and landmark 
        H_eb = HomogeneousTransformation(p[0:2],p[2])
        H_el.H = H_eb.H@lidar.H_bl.H  

        ##Plotting the non corner situation      
        # plot_2dframe(['observation','b','l','m'],[H_eb.H,H_el.H,z_lm],True)
        # plt.scatter(m_x, m_y,s=0.1)
        # plt.axis('equal')
        # plt.show()

for i in range(len(corner_training)):
    print('Entry:',i,', Class',corner_training[i].label, ', Size',corner_training[i].data_filled[:,0].size)
    print('Data 0:',corner_training[i].data_filled[:,0])
    print('Data 1',corner_training[i].data_filled[:,1])






##### Plotting Funcitons:##########
# # plot scan in the environment frame
# fig,ax = plt.subplots()
# show_scan(p, lidar, observation)
# ax.scatter(m_y, m_x,s=0.01)
# plt.show()

# # plot the raw sensor readings in polar coordinates
# plt.plot(np.rad2deg(observation[:, 1]), observation[:, 0], 'g.', label='Observations')
# plt.plot(np.rad2deg(observation[loc, 1]), observation[loc, 0], 'ro', label='Corner landmark')
# plt.xlabel('Bearing (sensor frame), degrees')
# plt.ylabel('Range, m')
# plt.show()

# # use the plot_2dframe function in plot_feeg6043
# plot_2dframe(['observation','b','l','m'],[H_eb.H,H_el.H,z_lm],True)
# plt.scatter(m_x, m_y,s=0.1)
# plt.axis('equal')
# plt.show()