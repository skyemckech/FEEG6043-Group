import numpy as np

# Define uncertainties (standard deviations)
sigma_d = 0.002  # Uncertainty in wheel diameter (meters)
sigma_W = 0.001  # Uncertainty in wheelbase (meters)
sigma_v = 0.05   # Uncertainty in wheel speed measurements (m/s)
sigma_w = 0.01   # Uncertainty in angular velocity (rad/s)

# Define wheel and actuator parameters
d = 0.07  # Wheel diameter in meters
W = 0.174  # Wheel spacing (track width) in meters

# Actuator configuration matrix G
G = np.array([
    [d / 4, d / 4],
    [-d / (2 * W), d / (2 * W)]
])

# Measurement uncertainty in wheel speed
R_w = np.array([
    [sigma_v**2, 0],
    [0, sigma_v**2]
])

# Compute Jacobian of G w.r.t. d and W
J_G = np.array([
    [1/4, 1/4],  # ∂G/∂d
    [-1/(2*W), 1/(2*W)]  # ∂G/∂W
])

# System parameter uncertainty matrix (diagonal since we assume no correlation)
R_d = np.diag([sigma_d**2, sigma_W**2])

# Compute propagated uncertainty in DOTX and DOTG
R_u = J_G @ R_d @ J_G.T + G @ R_w @ G.T

# Print the result for the uncertainty covariance matrix R_u
print("Uncertainty covariance matrix R_u (for [DOTX, DOTG]):")
print(R_u)

# Now define the propagate_uncertainties function
def propagate_uncertainties(N, E, G, v, w, dt, sigma_v, sigma_w):
    """
    Propagates the uncertainties in linear and angular velocity to the position (N, E) and heading (G).
    
    Arguments:
    - N: Current Northing (position in the north direction) in meters
    - E: Current Easting (position in the east direction) in meters
    - G: Current Heading (orientation in radians)
    - v: Linear velocity in m/s
    - w: Angular velocity in radians/s
    - dt: Time step in seconds
    - sigma_v: Uncertainty in linear velocity (m/s)
    - sigma_w: Uncertainty in angular velocity (rad/s)
    
    Returns:
    - mu_: New pose [N, E, G] after applying kinematics
    - sigma_N: Uncertainty in Northing
    - sigma_E: Uncertainty in Easting
    - sigma_G: Uncertainty in Heading
    """
    
    # Initialize new position and heading
    mu_ = np.zeros(3)
    
    # Jacobians for uncertainty propagation
    J_N_v = np.cos(G) * dt  # Jacobian of N with respect to v
    J_E_v = np.sin(G) * dt  # Jacobian of E with respect to v
    J_N_w = 0  # Jacobian of N with respect to w (no change in N from w)
    J_E_w = 0  # Jacobian of E with respect to w (no change in E from w)
    J_G_v = 0  # Jacobian of G with respect to v (no change in G from v)
    J_G_w = dt  # Jacobian of G with respect to w

    # For straight-line motion (w == 0)
    if w == 0:
        mu_[0] = N + v * J_N_v  # Update Northing (N)
        mu_[1] = E + v * J_E_v  # Update Easting (E)
        mu_[2] = G  # Heading remains the same
    else:
        # For motion with angular velocity (curved motion)
        mu_[0] = N + (v / w) * (np.sin(G + w * dt) - np.sin(G))  # Update Northing (N)
        mu_[1] = E - (v / w) * (np.cos(G + w * dt) - np.cos(G))  # Update Easting (E)
        mu_[2] = (G + w * dt) % (2 * np.pi)  # Update Heading (G) with wraparound
    
    # Calculate uncertainty propagation
    sigma_N = np.sqrt((J_N_v * sigma_v)**2 + (J_N_w * sigma_w)**2)  # Uncertainty in Northing
    sigma_E = np.sqrt((J_E_v * sigma_v)**2 + (J_E_w * sigma_w)**2)  # Uncertainty in Easting
    sigma_G = np.sqrt((J_G_v * sigma_v)**2 + (J_G_w * sigma_w)**2)  # Uncertainty in Heading

    # Combine the uncertainties in a single covariance matrix (5 states)
    R = np.zeros((5, 5))
    R[0, 0] = sigma_N**2  # Uncertainty in Northing
    R[1, 1] = sigma_E**2  # Uncertainty in Easting
    R[2, 2] = sigma_G**2  # Uncertainty in Heading
    R[3, 3] = R_u[0, 0]  # Uncertainty in wheel diameter (associated with linear velocity)
    R[4, 4] = R_u[1, 1]  # Uncertainty in wheelbase (associated with angular velocity)

    return mu_, R

# example use:
N = 0.0   # Initial Northing
E = 0.0   # Initial Easting
G = np.pi / 4  # Initial Heading (45 degrees)
v = 1.0   # Linear velocity in m/s
w = 0.1   # Angular velocity in rad/s
dt = 0.1  # Time step in seconds
sigma_v = 0.05  # Uncertainty in linear velocity
sigma_w = 0.01  # Uncertainty in angular velocity


# Propagate uncertainties and get the new state and uncertainty covariance
mu_, R = propagate_uncertainties(N, E, G, v, w, dt, sigma_v, sigma_w)

print("Updated Pose [N, E, G]:")
print(mu_)
print("5-State Uncertainty Covariance Matrix R:")
print(R)
