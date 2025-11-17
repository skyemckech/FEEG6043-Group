import numpy as np

from ..config import RobotConfig
from .PositionData import PositionData
from .UncertaintyData import UncertaintyData
from datetime import datetime
from zeroros.messages import LaserScan, Vector3Stamped, Pose, PoseStamped, Header, Quaternion

class ArucoBox:
    def __init__(self, config, simulation):
        self.config = config
        self.simulation = simulation
        self._position_data = None
        self._uncertainty_data = None
        self.timestamp = None


    @property
    def position_data(self):
        if self._position_data is None:
            self._position_data = PositionData()
        return self._position_data 
    
    @position_data.setter
    def position_data(self, msg):
        self.timestamp = msg.header.stamp
        self.position_data.northings = msg.pose.position.x
        self.position_data.eastings = msg.pose.position.y
        _, _, self.position_data.heading = msg.pose.orientation.to_euler()  
        self.position_data.wrap_angles()

    @property
    def uncertainty_data(self):
        if self._uncertainty_data == None:
            self._uncertainty_data = UncertaintyData()
            self._uncertainty_data.northings = self.config.aruco_northings_std
            self._uncertainty_data.eastings = self.config.aruco_eastings_std
        return self._uncertainty_data
    
    @uncertainty_data.setter
    def uncertainty_data(self, value):
        self._uncertainty_data = value
        
    def pose_parse(self, msg, sim_time_offset, aruco = False):
            # parser converts pose data to a standard format for logging
            time_stamp = msg[0]

            if aruco == True:
                # self.sim_time_offset is 0 if not a simulation. Deals with webots dealing in elapse timeself.sim_time_offset
                print(
                    "Received update from",
                    datetime.utcnow().timestamp() - msg[0] - sim_time_offset,
                    "seconds ago",
                )
                time_stamp = msg[0] + sim_time_offset                

            pose_msg = PoseStamped() 
            pose_msg.header = Header()
            pose_msg.header.stamp = time_stamp
            pose_msg.pose.position.x = msg[1]
            pose_msg.pose.position.y = msg[2]
            pose_msg.pose.position.z = 0

            quat = Quaternion()        
            if self.simulation == False and aruco == True: quat.from_euler(0, 0, np.deg2rad(msg[6]))
            else: quat.from_euler(0, 0, msg[6])
            pose_msg.pose.orientation = quat        
            return pose_msg