"""
Copyright (c) 2023 The uos_sess6072_build Authors.
Authors: Miquel Massot, Blair Thornton, Sam Fenton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""
import numpy as np
import argparse
import time
import openpyxl

from Libraries import *
from datetime import datetime
from drivers.aruco_udp_driver import ArUcoUDPDriver
from zeroros import Subscriber, Publisher
from zeroros.messages import LaserScan, Vector3Stamped, Pose, PoseStamped, Header, Quaternion
from zeroros.datalogger import DataLogger
from zeroros.rate import Rate
# from Libraries.model_feeg6043 import ActuatorConfiguration, rigid_body_kinematics, RangeAngleKinematics, feedback_control, TrajectoryGenerate, motion_model, extended_kalman_filter_predict, extended_kalman_filter_update
# from Libraries.math_feeg6043 import Vector, Inverse, HomogeneousTransformation, Identity, l2m, m2l, change_to_list, Matrix
# from Libraries.plot_feeg6043 import plot_zero_order,plot_trajectory,plot_2dframe
from matplotlib import pyplot as plt
from openpyxl import load_workbook
# add more libraries here
N = 0
E = 1
G = 2
DOTX = 3
DOTG = 4
class LaptopPilot:


    def __init__(self, simulation):

        self.config = RobotConfig(simulation)
        self.aruco_box = ArucoBox(self.config, simulation)
        self.aruco_driver = ArUcoUDPDriver(self.config.aruco_params, parent=self)
        print("Connecting to robot with IP", self.config.robot_ip)
        self.sim_time_offset = 0 #used to deal with webots timestamps
        self.sim_init = False #used to deal with webots timestamps
        if simulation:
            self.sim_init = True

        self.initialise_pose = True # False once the pose is initialised
        self.simulation = simulation

        self.ddrive = ActuatorConfiguration(self.config.wheel_distance, self.config.wheel_diameter) #look at your tutorial and see how to use this
        self.ControlAlgorithm = ControlAlgorithm(self.config, self.ddrive)
        self.ParticleFilter = ParticleFilter(self.config)
        # model pose
        self.est_pose_northings_m = None
        self.est_pose_eastings_m = None
        self.est_pose_yaw_rad = None

        #motion model variables
        self.state = None
        self.sensor_measurement = None

        # kalman filter
        self.jacobian = None
        self.covariance = None
        self.uncertainty_data_data = None

        #>Communication>#
        #################
    
        # measured pose
        self.measured_pose_timestamp_s = None
        self.measured_pose_northings_m = None
        self.measured_pose_eastings_m = None
        self.measured_pose_yaw_rad = None

        # wheel speed commands
        self.cmd_wheelrate_right = None
        self.cmd_wheelrate_left = None 

        # encoder/actual wheel speeds
        self.measured_wheelrate_right = None
        self.measured_wheelrate_right_next = None
        self.measured_wheelrate_left = None

        # measured ground speeds
        self.groundtruth_northings = None
        self.groundtruth_eastings = None
        self.groundtruth_yaw = None   

        # lidar
        self.lidar_timestamp_s = None
        self.lidar_data = None
        lidar_xb = 0 # location of lidar centre in b-frame primary axis ########################(changed)
        lidar_yb = 0.1 # location of lidar centre in b-frame secondary axis ###################(Changed)
        self.lidar = RangeAngleKinematics(lidar_xb,lidar_yb) ####################(changed)


        # Create variable for plotting ground truth and reference position
        self.p_reference_tracker = None
        self.p_groundtruth_tracker = None

        
        ###############################################################        

        self.datalog = DataLogger(log_dir="logs")

        # Wheels speeds in rad/s are encoded as a Vector3 with timestamp, 
        # with x for the right wheel and y for the left wheel.        
        self.wheel_speed_pub = Publisher(
            "/wheel_speeds_cmd", Vector3Stamped, ip=self.config.robot_ip
        )

        self.true_wheel_speed_sub = Subscriber(
            "/true_wheel_speeds",Vector3Stamped, self.true_wheel_speeds_callback,ip=self.config.robot_ip,
        )
        self.lidar_sub = Subscriber(
            "/lidar", LaserScan, self.lidar_callback, ip=self.config.robot_ip
        )
        self.groundtruth_sub = Subscriber(
            "/groundtruth", Pose, self.groundtruth_callback, ip=self.config.robot_ip
        )

    def true_wheel_speeds_callback(self, msg):
        print("Received sensed wheel speeds: R=", msg.vector.x,", L=", msg.vector.y)
        # update wheel rates
        self.measured_wheelrate_right = self.measured_wheelrate_right_next
        self.measured_wheelrate_right_next = msg.vector.x
        self.measured_wheelrate_left = msg.vector.y

        self.datalog.log(msg, topic_name="/true_wheel_speeds")

    def lidar_callback(self, msg):
        # This is a callback function that is called whenever a message is received        
        print("Received lidar message", msg.header.seq)        
        if self.sim_init == True:
            self.sim_time_offset = datetime.utcnow().timestamp()-msg.header.stamp
            self.sim_init = False     

        msg.header.stamp += self.sim_time_offset
        ###############(imported)#########################
        self.lidar_timestamp_s = msg.header.stamp #we want the lidar measurement timestamp here
        self.lidar_data = np.zeros((len(msg.ranges), 2)) #specify length of the lidar data
        self.lidar_data[:,0] = msg.ranges # use ranges as a placeholder, workout northings in Task 4
        self.lidar_data[:,1] = msg.angles # use angles as a placeholder, workout eastings in Task 4
        ###############(imported)#########################
        self.datalog.log(msg, topic_name="/lidar")

        # b to e frame
        p_eb = Vector(3)
        p_eb[0] = self.est_pose_northings_m #robot pose northings (see Task 3)
        p_eb[1] = self.est_pose_eastings_m #robot pose eastings (see Task 3)
        p_eb[2] = self.est_pose_yaw_rad #robot pose yaw (see Task 3)

        # m to e frame
        self.lidar_data = np.zeros((len(msg.ranges), 2))        
                    
        z_lm = Vector(2)        
        # for each map measurement
        for i in range(len(msg.ranges)):
            z_lm[0] = msg.ranges[i]
            z_lm[1] = msg.angles[i]
                
            t_em = self.lidar.rangeangle_to_loc(p_eb, z_lm) # see tutotial

            self.lidar_data[i,0] = t_em[0]
            self.lidar_data[i,1] = t_em[1]

        # this filters out any 
        self.lidar_data = self.lidar_data[~np.isnan(self.lidar_data).any(axis=1)]

    def groundtruth_callback(self, msg):
        """This callback receives the odometry ground truth from the simulator."""
        self.groundtruth_northings = msg.position.x
        self.groundtruth_eastings = msg.position.y 
        _, _, self.groundtruth_yaw = msg.orientation.to_euler()  
        self.datalog.log(msg, topic_name="/groundtruth")
    
    def aruco_update_and_log(self, aruco_pose):
        msg = self.aruco_box.pose_parse(aruco_pose, self.sim_time_offset, aruco = True)
        if self.sim_init == True:
                self.sim_time_offset = datetime.utcnow().timestamp()-msg.header.stamp
                self.sim_init = False 
            # converts aruco date to zeroros PoseStamped format
        
        self.datalog.log(msg, topic_name="/aruco")
        # update aruco data
        self.aruco_box.position_data = msg
        # reads sensed pose for local use
        self.measured_pose_northings_m = self.aruco_box.position_data.northings
        self.measured_pose_eastings_m = self.aruco_box.position_data.eastings
        self.measured_pose_yaw_rad = self.aruco_box.position_data.heading  

    def run(self, time_to_run=-1):
        self.start_time = datetime.utcnow().timestamp()
        
        try:
            r = Rate(10.0)
            while True:
                current_time = datetime.utcnow().timestamp()
                if time_to_run > 0 and current_time - self.start_time > time_to_run:
                    print("Time is up, stopping…")
                    break
                self.infinite_loop()
                r.sleep()
                
        except KeyboardInterrupt:
            print("KeyboardInterrupt received, stopping…")
        except Exception as e:
            print("Exception: ", e)
        finally:
            self.lidar_sub.stop()
            self.groundtruth_sub.stop()
            self.true_wheel_speed_sub.stop()
    
    def initialise_robot_pose(self):
        # Get initial position estimate
        if self.simulation:
            self.position_data = self.aruco_box.position_data
            self.uncertainty_data = self.aruco_box.uncertainty_data
        else:
            self.position_data = PositionData()
            self.uncertainty_data = UncertaintyData()
    
    def log_position_data(self):
        self.est_pose_northings_m = self.position_data.northings
        self.est_pose_eastings_m = self.position_data.eastings
        self.est_pose_yaw_rad = self.position_data.heading
        msg = self.aruco_box.pose_parse([datetime.utcnow().timestamp(),self.est_pose_northings_m,self.est_pose_eastings_m,0,0,0,self.est_pose_yaw_rad], self.sim_time_offset)
        self.datalog.log(msg, topic_name="/est_pose")

    def get_control_inputs(self):
        # feedforward control: check wp progress and sample reference trajectory
        p_ref, u_ref = self.path.p_u_sample(self.t) #sample the path at the current elapsetime (i.e., seconds from start of motion modelling)
        self.p_reference_tracker = p_ref[0:2,0]
        
        # feedback control: get pose change to desired trajectory from body
        dp = p_ref - self.position_data.position_vector #compute difference between reference and estimated pose in the $e$-frame
        # dp_truth = p_ref - p_robot_truth

        dp[2] = (dp[2] + np.pi) % (2 * np.pi) - np.pi # handle angle wrapping for yaw
        # dp_truth[2] = (dp_truth[2] + np.pi) % (2 * np.pi) - np.pi # handle angle wrapping for yaw

        H_eb = HomogeneousTransformation(self.position_data.position_vector[0:2], self.position_data.position_vector[2])
        error = Inverse(H_eb.H_R) @ dp # rotate the $e$-frame difference to get it in the $b$-frame (Hint: dp_b = H_be.H_R @ dp_e)

        return error, u_ref
    
    
    def send_wheel_commands(self, wheelspeeds):
        wheel_speed_msg = Vector3Stamped()
        wheel_speed_msg.vector.x = wheelspeeds[0,0] # Right wheelspeed rad/s
        wheel_speed_msg.vector.y = wheelspeeds[1,0] # Left wheelspeed rad/s

        self.cmd_wheelrate_right = wheel_speed_msg.vector.x
        self.cmd_wheelrate_left = wheel_speed_msg.vector.y

        self.wheel_speed_pub.publish(wheel_speed_msg)
        self.datalog.log(wheel_speed_msg, topic_name="/wheel_speeds_cmd")
        
    def infinite_loop(self):
        """Main control loop

        Your code should go here.
        """
        # > Sense < #
        # get the latest position measurements
        aruco_pose = self.aruco_driver.read()    

        if aruco_pose is not None:
            self.aruco_update_and_log(aruco_pose)

            # initialisation step
            if self.initialise_pose == True:
                # set initial measurements
                self.initialise_robot_pose()
                self.ParticleFilter.initialise_position(self.position_data, self.uncertainty_data)
                self.path = generate_trajectory(self.config, self.position_data)
                
                # get current time and determine timestep
                self.t_prev = datetime.utcnow().timestamp() #initialise the time
                self.t = 0 #elapsed time
                time.sleep(0.1) #wait for approx a timestep before proceeding
                
                # path and tragectory are initialised
                self.initialise_pose = False

        if self.initialise_pose != True:  
            
             # > Receive < #
            #################################################################################
            # convert true wheel speeds in to twist
            q = Vector(2)            
            q[0] = self.measured_wheelrate_right # wheel rate rad/s (measured)
            q[1] = self.measured_wheelrate_left # wheel rate rad/s (measured)
            u = self.ddrive.fwd_kinematics(q)    
            
            #determine the time step
            t_now = datetime.utcnow().timestamp()        
                    
            dt = t_now - self.t_prev #timestep from last estimate
            self.t += dt #add to the elapsed time
            self.t_prev = t_now #update the previous timestep for the next loop

             # > Think < #
            ################### Motion Model ##############################
            # take current pose estimate and update by twist
            self.ParticleFilter.update_estimate(self.position_data, u, dt)

            if aruco_pose is not None:
                self.ParticleFilter.apply_measurement(self.aruco_box)
        
            self.position_data.position_vector = self.ParticleFilter.get_new_estimate()

            #################### Trajectory sample #################################    
            self.path.wp_progress(self.t, self.position_data.position_vector,self.config.accept_radius,2,self.config.timeout) # fill turning radius
            # update for show_laptop.py
            self.log_position_data()
            # > Control < #
            ################################################################################
            error, u_ref = self.get_control_inputs()
            wheelspeeds = self.ControlAlgorithm.apply_control(error, u_ref)
            self.send_wheel_commands(wheelspeeds)
            ################################################################################
            # > Act < #
            # Send commands to the robot        
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--time",
        type=float,
        default=-1,
        help="Time to run an experiment for. If negative, run forever.",
    )
    parser.add_argument(
        "--simulation",
        action="store_true",
        help="Run in simulation mode. Defaults to False",
    )

    args = parser.parse_args()

    laptop_pilot = LaptopPilot(args.simulation)
    laptop_pilot.run(args.time)