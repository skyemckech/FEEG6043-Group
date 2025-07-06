import numpy as np

class RobotConfig:
    def __init__(self, simulation):
        # network for sensed pose
        self.aruco_params = {
            "port": 50000,  # Port to listen to (DO NOT CHANGE)
            "marker_id": 22,  # Marker ID to listen to (CHANGE THIS to your marker ID)            
        }
        self.robot_ip = "192.168.90.1"
        
        # handles different time reference, network amd aruco parameters for simulator

        if simulation:
            self.robot_ip = "127.0.0.1"          
            self.aruco_params['marker_id'] = 0  #Ovewrites Aruco marker ID to 0 (needed for simulation)
            self.sim_init = True #used to deal with webots timestamps
        
        # path
        self.path_velocity = 0.05
        self.path_acceleration = 0.1/3
        self.path_radius = 0.3
        self.accept_radius = 0.2
        lapx = [0,1.5,1.5,0]
        lapy = [0,0,1.5,1.5]
        self.northings_path = lapx+[0]
        self.eastings_path = lapy+[0]      
        self.relative_path = True #False if you want it to be absolute  
        # modelling parameters
        self.wheel_distance = 0.174/2 # m 
        self.wheel_diameter = 0.070 # m
        
        # control parameters        
        self.tau_s = 2 # s to remove along track error
        self.L = 0.4 # m distance to remove normal and angular error
        self.v_max = 0.2 # m/s fastest the robot can go
        self.w_max = np.deg2rad(30) # fastest the robot can turn
        self.timeout = 10 #s

        # Particle filter parameters
        self.g_std   = np.deg2rad(1)   # rad
        self.x_dot_std = 0.3 #m/s
        self.g_dot_std = np.deg2rad(0.1) #rad/s
        self.aruco_northings_std = 0.01 #m
        self.aruco_eastings_std = 0.01 #m
        self.N = 100
