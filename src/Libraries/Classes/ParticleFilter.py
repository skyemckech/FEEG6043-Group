import numpy as np
import copy

from ..model_feeg6043 import Particles, initialise_particle_distribution, discrete_motion_model, Measurement, pf_update, pf_resample, neff, kde_probability
from ..math_feeg6043 import wrapped_mean, Vector
from ..config import RobotConfig

class ParticleFilter:
    def __init__(self, config):
        self.config = config
        
        # Initialise PF 
        self.particles = Particles(self.config.N)

        # Initialise params
        self.sigma_resolution = None
        self.sampling_resolution = self.particles.N #samples over northings and easting range respectively
        
    def initialise_position(self, PositionData, UncertaintyData):
        initialise_particle_distribution(self.particles, centre=[PositionData.northings, PositionData.eastings], radius = np.sqrt(4*UncertaintyData.northings*UncertaintyData.eastings))
        self.particles.gamma = [PositionData.heading]*self.particles.N

    def update_estimate(self, PositionData, u, dt):
        self.particles = discrete_motion_model(self.particles, PositionData.heading, u, dt, [self.config.g_std, self.config.x_dot_std, self.config.g_dot_std])
    
    def apply_measurement(self, ArucoBox):

        measurement = Measurement()
        # use these measurement for update prediction
        measurement.timestamp = ArucoBox.timestamp
        measurement.northings = ArucoBox.position_data.northings
        measurement.northings_std = ArucoBox.uncertainty_data.northings
        measurement.eastings = ArucoBox.position_data.eastings
        measurement.eastings_std = ArucoBox.uncertainty_data.eastings

        self.particles = pf_update(self.particles, measurement)
        
        self.sigma_resolution = np.sqrt(measurement.northings_std*measurement.eastings_std)

        # resample if needed
        pf_resample(self.particles, self.sigma_resolution, verbose=True)

        # store particles when there is a measurement 
        
    def get_new_estimate(self):
        
        est_northings, est_eastings = kde_probability(self.particles, self.sigma_resolution, self.sampling_resolution)
        est_heading = wrapped_mean(self.particles.gamma)

        return np.array([[est_northings], [est_eastings], [est_heading]])
    