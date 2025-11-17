import numpy as np

from ..math_feeg6043 import Vector
from dataclasses import dataclass


class PositionData:
    def __init__(self):
        '3 value vector for position data'
        self._position_vector = Vector(3)
    ## Properties
    ###############################
    @property
    def northings(self):
        northings = self._position_vector[0]
        return float(northings)
    
    @northings.setter
    def northings(self, value):
        self._position_vector[0] = value
    
    @property
    def eastings(self):
        eastings = self._position_vector[1]
        return float(eastings)
    
    @eastings.setter
    def eastings(self, value):
        self._position_vector[1] = value
    
    @property
    def heading(self):
        heading = self._position_vector[2]
        return float(heading)
    
    @heading.setter
    def heading(self, value):
        self._position_vector[2] = value
    
    @property
    def position_vector(self):
        position_vector = self._position_vector
        return position_vector
    
    @position_vector.setter
    def position_vector(self, vector):
        self._position_vector = vector
    
    ## Methods
    ####################################
    
    def wrap_angles(self):
        self._position_vector[2] = self._position_vector[2] % (np.pi*2) # manage angle wrapping