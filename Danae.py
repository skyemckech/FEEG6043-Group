from Libraries.plot_feeg6043 import plot_kalman
from Libraries.model_feeg6043 import extended_kalman_filter_predict, extended_kalman_filter_update
import numpy as np

class EKF:
    
    def __init__(self):
        
            Sigma = np.eye(len(initial_state))
            mu = np.array(initial_state)
            zm = Matrix (1,1)
            print("zm",zm)
            print("mu",mu)
            print("Sigma",Sigma)


            ##CREATING THE BASE MATRIX FOR Q,U AND R!!!!!########     
            om = Matrix(1,1); om[0,0] =1

            ### ADDED CONTROL AND PROCESS NOISE!!!!!!!!!########
            Q = om #measurement noise
            u = 2*om #control noise
            R = 2*om #process noise 
            state = zm; 
            covariance = om
            wrapping_index = True