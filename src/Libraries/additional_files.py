import numpy as np
from .math_feeg6043 import Vector,Matrix,Identity,Transpose,Inverse,v2t,t2v,HomogeneousTransformation, interpolate, short_angle, inward_unit_norm, line_intersection, cartesian2polar, polar2cartesian
from .plot_feeg6043 import plot_kalman, plot_graph, show_information
from .model_feeg6043 import rigid_body_kinematics

def m2l(m): return [element for row in m for element in row]

def change_to_list(data):
    if type(data) == type(np.array([])):
        return m2l(data)
    if type(data) == type(2.0):
        return [data]
    if type(data) == type([]):
        return data
    if type(data) == type("string"):
        return [data]
    else:
        return TypeError
    
def motion_model(state, u, dt):

    N = 0
    E = 1
    G = 2
    DOTX = 3
    DOTG = 4

    N_k_1 = state[N]
    E_k_1 = state[E]
    G_k_1 = state[G]
    DOTX_k_1 = state[DOTX]
    DOTG_k_1 = state[DOTG]

    p = Vector(3)
    p[0] = N_k_1
    p[1] = E_k_1
    p[2] = G_k_1
    
    # note rigid_body_kinematics already handles the exception dynamics of w=0
    p, _,_,_ = rigid_body_kinematics(p,u,dt)    

    # vertically joins two vectors together
    state = np.vstack((p, u))
    
    N_k = state[N]
    E_k = state[E]
    G_k = state[G]
    DOTX_k = state[DOTX]
    DOTG_k = state[DOTG]
    
    # Compute its jacobian
    F = Identity(5)    
    
    if abs(DOTG_k) <1E-2: # caters for zero angular rate, but uses a threshold to avoid numerical instability
        F[N, G] = -DOTX_k * dt * np.sin(G_k_1)
        F[N, DOTX] = dt * np.cos(G_k_1)
        F[E, G] = DOTX_k * dt * np.cos(G_k_1)
        F[E, DOTX] = dt * np.sin(G_k_1)
        F[G, DOTG] = dt
        
    else:
        F[N, G] = (DOTX_k/DOTG_k)*(np.cos(G_k)-np.cos(G_k_1))
        F[N, DOTX] = (1/DOTG_k)*(np.sin(G_k)-np.sin(G_k_1))
        F[N, DOTG] = (DOTX_k/(DOTG_k**2))*(np.sin(G_k_1)-np.sin(G_k))+(DOTX_k*dt/DOTG_k)*np.cos(G_k)
        F[E, G] = (DOTX_k/DOTG_k)*(np.sin(G_k)-np.sin(G_k_1))
        F[E, DOTX] = (1/DOTG_k)*(np.cos(G_k_1)-np.cos(G_k))
        F[E, DOTG] = (DOTX_k/(DOTG_k**2))*(np.cos(G_k)-np.cos(G_k_1))+(DOTX_k*dt/DOTG_k)*np.sin(G_k)
        F[G, DOTG] = dt

    return state, F