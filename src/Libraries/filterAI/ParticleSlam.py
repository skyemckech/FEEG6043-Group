from .PositionData import PositionData
from ..model_feeg6043 import RangeAngleKinematics, ParticlePathSLAM


class ParticleSlam:
    def __init__(self, config):
        self.config = config

    def initialise_position(self, position: PositionData, lidar: RangeAngleKinematics):
        particle_filter = ParticlePathSLAM(self.config.N,
                                 lidar,
                                 position, 
                                 self.config.position_std,                                 
                                 self.config.auxiliary_noise)
        
    