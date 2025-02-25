
from Tutorials.math_feeg6043 import Matrix
import numpy as np
import matplotlib.pyplot as plt

class EKF:
    def __init__(self, initial_state):
        
        return

    def set_parameters(self, Q_factor, u_factor, R_factor):
        # define a zero matrix (zm) and a one matrix (om)
        om = Matrix(1,1); om[0,0] =1
        zm = Matrix(1,1)

        # set previous belief and timestep
        self.state = zm; self.covariance = om
        self.dt = 1

        # set control and process noise
        self.u = u_factor*om
        self.R = R_factor*om

        # set measurement noise
        self.z = -2*om

        self.Q = Q_factor*om
        
        return self.Q, self.u, self.R, self.z, self.state, self.covariance, self.dt
    

    def f_nonlintest(x, u, dt):
            # this is a non-linear model that cannot be solved with a KF f(x)=x**2+u
            F = np.zeros((1, 1), dtype=float)
            F[0,0] = 2*x
            return x ** 2 + u, F

        
    def h(x):
        H = Matrix(1,1)
        H[0,0] = 1
        return x, H
    
    def extended_kalman_filter_predict(mu, Sigma, u, f, R, dt):
    # (1) Project the state forward (f = rigid body motion model)
        pred_mu, F = f(mu, u, dt)
      
    # (2) Project the error forward: R is covancerance
        pred_Sigma = (F @ Sigma @ F.T) + R
    
    # Return the predicted state and the covariance
        return pred_mu, pred_Sigma

    def extended_kalman_filter_update(mu, Sigma, z, h, Q, wrap_index = None):
        
        ##---Prepare the estimated measurement-----
        pred_z, H = h(mu)
    
        ####Compute the Kalman gain####
        K = Sigma @ H.T @ np.linalg.inv(H @ Sigma @ H.T + Q)
        
        # ###Compute the updated state estimate #####
        delta_z = z- pred_z        
        if wrap_index != None: delta_z[wrap_index] = (delta_z[wrap_index] + np.pi) % (2 * np.pi) - np.pi    
        cor_mu = mu + K @ (delta_z)

        # (5) Compute the updated state covariance
        cor_Sigma = (np.eye(mu.shape[0], dtype=float) - K @ H) @ Sigma
        
        # Return the state and the covariance
        return cor_mu, cor_Sigma
    
    # def kalman_filter_process(self,state, covariance, u, f_nonlin, R, dt , z , h , Q , view_flag=True):


    #     self.pred_state, self.pred_covariance = extended_kalman_filter_predict(state, covariance, u, f_nonlin, R, dt,view_flag=True)
    #     print('Time predicted is', dt, 's', 'control predicted is', u, 'state predicted is', cor_state, 'covariance predicted is', cor_covariance)
    #     # cor_state, cor_covariance = extended_kalman_filter_update(pred_state, pred_covariance,z,h,Q,view_flag=True)
    #     # print('Time is', dt, 's', 'control is', u, 'state is', cor_state, 'covariance is', cor_covariance)

    #     return self.pred_state, self.pred_covariance 

testexample = EKF
Q_factor= 2 
u_factor = 2 
R_factor = 2
Q, u, R, z, state, covariance, dt = testexample.set_noise(Q_factor, u_factor, R_factor)

or_state, cor_covarianc = testexample.kalman_filter_process(state, covariance, u, testexample.f_nonlintest, R, dt , z , testexample.h , Q , view_flag=True)
print(cor_state,cor_covarianc)