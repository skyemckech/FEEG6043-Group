from ..model_feeg6043 import feedback_control

class ControlAlgorithm:
    def __init__(self, config, ddrive):
        self.config = config
        self.ddrive = ddrive
        self.k_n = 0 #kn
        self.k_g = 0 #kg
        self.k_s = None

    def apply_control(self, error, u_ref):
        # compute control gains for the initial condition (where the robot is stationalry)
        self.k_s = 1/self.config.tau_s #ks
        
        # update the controls
        du = feedback_control(error, self.k_s, self.k_n, self.k_g)

        # total control
        #u = u_ref + du # combine feedback and feedforward control twist components
        u = u_ref + du

        # update control gains for the next timestep
        self.k_n = 2*u[0]/(self.config.L**2) #kn
        self.k_g = u[0]/self.config.L #kg

        # ensure within performance limitation
        if u[0] > self.config.v_max: u[0] = self.config.v_max
        if u[0] < -self.config.v_max: u[0] = -self.config.v_max
        if u[1] > self.config.w_max: u[1] = self.config.w_max
        if u[1] < -self.config.w_max: u[1] = -self.config.w_max

        # actuator commands                 
        q = self.ddrive.inv_kinematics(u) 

        return q