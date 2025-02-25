
from Libraries.plot_feeg6043 import plot_kalman
from Libraries.model_feeg6043 import extended_kalman_filter_predict, extended_kalman_filter_update
import numpy as np
import matplotlib.pyplot as plt

class EKF:
    def __init__(self):
        
        Sigma = np.eye(len(initial_state))
        mu = np.array(initial_state)
        zm = Matrix (1,1)



        ##CREATING THE BASE MATRIX FOR Q,U AND R!!!!!########     
        om = Matrix(1,1); om[0,0] =1
     
            ### ADDED CONTROL AND PROCESS NOISE!!!!!!!!!########
        Q = om #measurement noise
        u = 2*om #control noise
        R = 2*om #process noise 
        state = zm; 
        covariance = om
        wrapping_index = True




    def f_nonlintest(x, u, dt):
            # this is a non-linear model that cannot be solved with a KF f(x)=x**2+u
            F = np.zeros((1, 1), dtype=float)
            #F[0,0] = 2*x
            return x ** 2 + u , F

            ##### PLOT TEST #####

            # Generate x values for plotting
            
    x_values = np.linspace(-5, 5, 100)  # Range from -5 to 5
    u = 1  # Example input
    dt = 1  # Not used in this function

            # Compute f(x) for each x in x_values
    f_values = [f_nonlintest(x_values, u, dt)]

            # Plot the function
    print(f_values)
    # plt.figure(figsize=(8, 5))
    # plt.plot(x_values, f_values, label=r'$f(x) = x^2 + u$', color='b')
    # plt.xlabel('x')
    # plt.ylabel('f(x)')
    # plt.title('Non-Linear Function Plot')
    # plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    # plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
    # plt.legend()
    # plt.grid()
    # plt.show()

        # def h(x):
        #     H = Matrix(1,1)
        #     H[0,0] = 1
        #     return x, H
    
    # def kalman_filter_process(self,state, covariance, u, f_nonlin, R, dt , z , h , Q , view_flag=True)

    # pred_state, pred_covariance = extended_kalman_filter_predict(state, covariance, u, f_nonlin, R, dt,view_flag=True)

    # print('Time predicted is', dt, 's', 'control predicted is', u, 'state predicted is', cor_state, 'covariance predicted is', cor_covariance)

    # cor_state, cor_covariance = extended_kalman_filter_update(pred_state, pred_covariance,z,h,Q,view_flag=True)

    # print('Time is', dt, 's', 'control is', u, 'state is', cor_state, 'covariance is', cor_covariance)

    # return cor_state, cor_covariance