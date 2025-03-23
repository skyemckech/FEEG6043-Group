import numpy as np
import copy
from matplotlib import pyplot as plt
from Libraries.math_feeg6043 import Vector,Matrix,Identity,Transpose,Inverse,v2t,t2v,HomogeneousTransformation, interpolate, short_angle, inward_unit_norm, line_intersection, cartesian2polar, polar2cartesian
from Libraries.plot_feeg6043 import plot_kalman
import matplotlib.pyplot as plt
#from Libraries.model_feeg6043 import rangeangle_to_loc

x_bl = 0; y_bl = 0
t_bl = Vector(2)
t_bl[0] = x_bl
t_bl[1] = y_bl
H_bl = HomogeneousTransformation(t_bl,0)
H_eb_ = HomogeneousTransformation()
p = Vector(3); 
p[0] = 0 #Northings
p[1] = 0 #Eastings
#p[2] = np.deg2rad(0) #Heading (rad)


def find_corner(corner_data, threshold = 0.01):
    # identify the reference coordinate as the inflection point
    
    # Step 1: Compute slope
    slope = np.gradient(corner_data[:, 0])
    
    # Step 2: Compute the second derivative (curvature)
    curvature = np.gradient(slope)
    
    # Step 3: Check if criteria is more than threshold    
    print('Max inflection value is ',np.nanmax(abs(np.gradient(np.gradient(curvature)))), ': Threshold ',threshold)
    if np.nanmax(abs(np.gradient(np.gradient(curvature)))) > threshold:
        # compute index of inflection point    
        largest_inflection_idx = np.nanargmax(abs(np.gradient(np.gradient(curvature))))
        
        r = corner_data[largest_inflection_idx, 0]  # Radial distance at the largest curvature
        theta = corner_data[largest_inflection_idx, 1]  # Angle at the largest curvature
        return r, theta, largest_inflection_idx
    
    else:
        return None, None, None  # No inflection points found



def rangeangle_to_loc(p_eb, z_lm):
        
        # generate a homogeneous transformation of the robot pose
        H_eb = HomogeneousTransformation(p_eb[0:2],p_eb[2])
        
        # check that the sensor readings provided are possible
        #z_lm = check_range(z_lm)
        
        # convert the sensor observation in from polar to cartesian coordinates        
        r = z_lm[0]
        theta = z_lm[1]
        
        
        t_lm = polar2cartesian(r,theta)
        
        # Apply the homogeneous transformation to get from the sensor frame to the body with H_bl, and then to the fixed frame H_eb
        t_em = t2v(H_eb.H@ H_bl.H @v2t(t_lm))




        return t_em


def find_cuner(scan_data, threshold = 0.01):
      
    scan_cartizie = np.zeros((len(scan_data),2))
    scan_grad = np.zeros((len(scan_data)-1,2))
    final_value = len(scan_data)

    #use blairs function to give an inital guess of the corner:
    r, theta, largest_inflection_idx = find_corner(scan_data, threshold)
    infection_point_polar = np.array([r, theta])

    ##converts this to cartisian:
    infection_point_cart = rangeangle_to_loc(p, infection_point_polar)


    # converts sample to cartisian:
    for i in range(len(scan_data)):

        scan_cartizie[i] = rangeangle_to_loc(p,scan_data[i])
    
   # plots data and example point;
    print("scan_cartizie:",scan_cartizie)
    plt.plot(scan_cartizie[:,0], scan_cartizie[:,1], marker='o', linestyle='-', color='b', label="Line Plot")
    plt.plot(infection_point_cart[0], infection_point_cart[1], marker='o', linestyle='-', color='r', label="Line Plot")
    # Labels & Title
    plt.xlabel("X-Axis")
    plt.ylabel("Y-Axis")
    plt.title("Simple Line Plot")
    plt.legend()  # Show legend
    plt.show()
         
    return scan_data


#bot_pose = [0,0,0]
sample_scan = np.array([
    [1.18002099, 0.10605896],
    [1.18773236, 0.10628763],
    [1.19562772, 0.10652183],
    [1.20371884, 0.10676171],
    [1.21201807, 0.10700786],
    [1.22053887, 0.10726058],
    [1.22929562, 0.1075203 ],
    [1.2383039,  0.10778743],
    [1.24758031, 0.10806257],
    [1.25714304, 0.10834614],
    [1.26701161, 0.10863877],
    [1.27720721, 0.10894114],
    [1.28775285, 0.10925395],
    [1.29869819, 0.10955566],
    [1.3099256,  0.10997662],
    [1.32170072, 0.11030708],
    [1.33395302, 0.11064298],
    [1.3467118,  0.11099171],
    [1.36000334, 0.11136505],
    [1.37374779, 0.11185433],
    [1.38831788, 0.11222126],
    [1.40331266, 0.11279241],
    [1.41939023, 0.11315726],
    [1.43591391, 0.11378718],
    [1.45376567, 0.11416769],
    [1.47216312, 0.11483853],
    [1.49209652, 0.11527108],
    [1.51280177, 0.11596948],
    [1.53466689, 0.11677335],
    [1.55872476, 0.11723756],
    [1.58377136, 0.11805999],
    [1.61047574, 0.11898908],
    [1.64003665, 0.11960076],
    [1.67123956, 0.12055551],
    [1.70488815, 0.12162669],
    [1.74131746, 0.12282373],
    [1.78092498, 0.12415851],
    [1.82600758, 0.12504491],
    [1.87434323, 0.12647522],
    [1.94098109, 0.17854281],
    [1.94045005, 0.19645111],
    [1.9399235,  0.21420471],
    [1.9394011,  0.23181626],
    [1.93888272, 0.24929803],
    [1.93836766, 0.26666205],
    [1.93785582, 0.28391994],
    [1.9373469,  0.30108309],
    [1.9368403,  0.31816269],
    [1.93633587, 0.33516966],
    [1.93583338, 0.35211477],
    [1.93533232, 0.36900862],
    [1.9348325,  0.3858617 ],
    [1.93433367, 0.40268439],
    [1.93383532, 0.41948701],
    [1.93333707, 0.4362798 ],
    [1.93283914, 0.45307302],
    [1.93234091, 0.46987689],
    [1.93184187, 0.48670167],
    [1.93134206, 0.5035577 ],
    [1.93084078, 0.52045531],
    [1.93033833, 0.53740509],
    [1.92983362, 0.55441755],
    [1.92932689, 0.57150353],
    [1.92881766, 0.588674  ],
    [1.92830556, 0.60594009],
    [1.92779037, 0.62331326],
    [1.9272716,  0.64080516],
    [1.92674888, 0.65842778],
    [1.92622206, 0.6761935 ],
    [1.92569063, 0.69411502]
])

#lidar_test_data = np.zeros((50, 2))
#print(lidar_test_data)
#lidar.rangeangle_to_loc
#print(sample_scan[:,1])
#first_sample = Vector(2)
#first_sample = sample_scan[]
#print(sample_scan[0])



a = find_cuner(sample_scan)





# sample_scan_cart = np.zeros((len(sample_scan),2))
# # converts first sample to cartisian:
# for i in range(len(sample_scan)):
#     sample_scan_cart[i] = rangeangle_to_loc(p,sample_scan[i])

# #print(len(sample_scan_cart))
# #print(sample_scan[:,0])
# #r, theta, largest_inflection_idx = find_corner(sample_scan_cart)
# r, theta, largest_inflection_idx = find_corner(sample_scan)

# infection_point_polar = np.array([r, theta])
# print("infection_point_polar:",infection_point_polar)

# infection_point_cart = rangeangle_to_loc(p, infection_point_polar)

# print("infection_point_cart:",infection_point_cart)
# print("sample_scan_cart",sample_scan_cart)


# print("radis of the infleciton point:", r )
# print("angle of the infleciton point:", theta )
# print("largest_inflection_idx of the infleciton point:", largest_inflection_idx )

# plt.plot(sample_scan_cart[:,0], sample_scan_cart[:,1], marker='o', linestyle='-', color='b', label="Line Plot")
# plt.plot(infection_point_cart[0], infection_point_cart[1], marker='o', linestyle='-', color='r', label="Line Plot")
# # Labels & Title
# plt.xlabel("X-Axis")
# plt.ylabel("Y-Axis")
# plt.title("Simple Line Plot")
# plt.legend()  # Show legend
# plt.show()