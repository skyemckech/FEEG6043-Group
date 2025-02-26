from Libraries.math_feeg6043 import Matrix
from Libraries.model_feeg6043 import extended_kalman_filter_predict, extended_kalman_filter_update
import numpy as np
import matplotlib.pyplot as plt

class EKF:
    def __init__(self, initial_state):
        if initial_state is not None:
            self.state = initial_state
        else:
            self.state = None
        self.covariance = None

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

        return self.Q , self.u, self.R, self.z, self.state, self.covariance, self.dt

    def f_nonlintest(self, x, u, dt):
            # this is a non-linear model that cannot be solved with a KF f(x)=x**2+u
            F = np.zeros((1, 1), dtype=float)
            F[0,0] = 2*x
            return x ** 2 + u, F

        
    def h(self, x):
        H = Matrix(1,1)
        H[0,0] = 1
        return x, H
    

testexample = EKF(None)
Q_factor= 1 ###measurment nosie
u_factor = 2 ###control nosie
R_factor = 2  ##process nosie
Q, u, R, z, state, covariance, dt = testexample.set_parameters(Q_factor, u_factor, R_factor)
# print(Q, u, R, z, state, covariance, dt)

pred_mu, pred_Sigma = extended_kalman_filter_predict(state, covariance, u, testexample.f_nonlintest, R, dt,view_flag=False)
print(pred_mu, pred_Sigma)
cor_mu, cor_Sigma = extended_kalman_filter_update(pred_mu, pred_Sigma, z, testexample.h, Q,view_flag=False)
print(cor_mu, cor_Sigma)