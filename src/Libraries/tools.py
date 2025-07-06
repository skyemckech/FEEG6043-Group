from .math_feeg6043 import l2m
from .model_feeg6043 import TrajectoryGenerate

def generate_trajectory(config, robot_position):
    # pick waypoints as current pose relative or absolute northings and eastings
        if config.relative_path == True:
            for i in range(len(config.northings_path)):
                config.northings_path[i] += robot_position.northings #offset by current northings
                config.eastings_path[i] += robot_position.eastings #offset by current eastings

        # convert path to matrix and create a trajectory class instance
        C = l2m([config.northings_path, config.eastings_path])        
        path = TrajectoryGenerate(C[:,0],C[:,1])        
            
        # set trajectory variables (velocity, acceleration and turning arc radius)
        path.path_to_trajectory(config.path_velocity, config.path_acceleration) #velocity and acceleration
        path.turning_arcs(config.path_radius) #turning radius
        path.wp_id=0 #initialises the next waypoint
        
        return path