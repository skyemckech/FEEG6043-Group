import numpy as np
import copy
from matplotlib import pyplot as plt
from Libraries.math_feeg6043 import Vector,Matrix,Identity,Transpose,Inverse,v2t,t2v,HomogeneousTransformation, interpolate, short_angle, inward_unit_norm, line_intersection, cartesian2polar, polar2cartesian
from Libraries.plot_feeg6043 import plot_kalman
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

x_bl = 0; y_bl = 0
t_bl = Vector(2)
t_bl[0] = x_bl
t_bl[1] = y_bl
H_bl = HomogeneousTransformation(t_bl,0)
H_eb_ = HomogeneousTransformation()
p = Vector(3); 
p[0] = 1 #Northings
p[1] = 2 #Eastings


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

def loc_to_rangeangle(p_eb, t_em):

    # generate a homogeneous transformation of the robot pose
    H_eb = HomogeneousTransformation(p_eb[0:2],p_eb[2])

    # get the location of the map feature in the body frame using Inverse(H_eb)=H_be, and then to from the body to the sensor frame using Inverse(H_bl)=H_lb
    t_lm = t2v(Inverse(H_bl.H)@Inverse(H_eb.H)@v2t(t_em))

    # Convert the cartesian vector t_lm in the sensor frame to the equivalent polar coordinates
    r,theta = cartesian2polar(t_lm[0],t_lm[1])

    z_lm = Vector(2)
    z_lm[0] = r
    z_lm[1] = theta   
    
    # check that the sensor readings provided are possible
    #z_lm = check_range(z_lm)        

    return z_lm


def find_cuner(scan_data,bot_pose_cart, threshold = 0.01, corner_likeness = 10, size_limit = 10):
      ### takes cartisian as an input:
    scan_data_cart = np.zeros((len(scan_data),2))
    hype_array = np.zeros((len(scan_data))) ####  hype_containter = np.zeros((len(scan_data,1))) maybe

    j = 0

                                                                                                                        
    ##Convierts sample to polar:
    for i in range(len(scan_data)):
        scan_data_cart[i] = rangeangle_to_loc(p,scan_data[i])



    #use blairs function to give an inital guess of the corner (from cart to polar):
    r, theta, largest_inflection_idx = find_corner(scan_data, threshold)
    infection_point_polar = np.array([r, theta])
    ##converts this to cartisian:
    infection_point_cart = rangeangle_to_loc(bot_pose_cart, infection_point_polar)

    

    #1) Finds the closest point to the inflextion point, and set a limit to say data has to be somewhat good for use to make a dessition:
    for i in range(len(scan_data)):
         
         scan_differnece = scan_data_cart[i] - infection_point_cart
         hype_current = (scan_differnece[0])**2 + (scan_differnece[1])**2 
         hype_array[i] = hype_current 
         
    ##1.1 make sure data point if close to the one that is furtherst away. protect against strange conditions...

    hype_min_value = np.min(hype_array)  # Get the smallest value
    hype_min_value_index = np.argmin(hype_array) #get the possition of the smallest value
    max_value = np.max(scan_data[:,0])  # Get the largest value
    max_index = np.argmax(scan_data[:,0])  # Get the index of the largest value

    if corner_likeness < hype_min_value:
         
         return print("not a corner not enough data around coner location, hype_min_value:",hype_min_value)


    # Find local minima
    local_maximia_indices = argrelextrema(scan_data[:,0], np.greater)[0]

    print("infection_point_cart:",infection_point_cart)
    print("infection_point_cart[0]:",infection_point_cart[0])
    print("local Maxima:",local_maximia_indices)
    print(" max_index:",max_index)
    print("max_value:",max_value)
    print("hype_min_value_index:",hype_min_value_index)
    print("hype_min_value:",hype_min_value)
    print("hype_array:",hype_array)




    # if size_limit < hype_array:
         
    #      return print("not a corner size limit exeeded: index __")
    
    # elif :




    #2) Take first and last 5 from each side and compaire grads to get and spread of data
    ###  2.1) use data to find range of uncertainties for shape// overall slope// angle of the wall




    
   # plots data and example point;
    #print("scan_cartizie:",scan_cartizie)
    plt.plot(scan_data[:,0], scan_data[:,1], marker='o', linestyle='-', color='b', label="Line Plot")
    plt.plot(infection_point_polar[0], infection_point_polar[1], marker='o', linestyle='-', color='r', label="Line Plot")
    # Labels & Title
    plt.xlabel("X-Axis")
    plt.ylabel("Y-Axis")
    plt.title("Simple Line Plot")
    plt.legend()  # Show legend
    plt.show()

    plt.plot(scan_data_cart[:,0], scan_data_cart[:,1], marker='o', linestyle='-', color='b', label="Line Plot")
    plt.plot(infection_point_cart[0], infection_point_cart[1], marker='o', linestyle='-', color='r', label="Line Plot")
    # Labels & Title
    plt.xlabel("X-Axis")
    plt.ylabel("Y-Axis")
    plt.title("Simple Line Plot")
    plt.legend()  # Show legend
    plt.show()

    plt.plot(range(len(scan_data)),scan_data[:,0], marker='o', linestyle='-', color='b', label="Line Plot")
    plt.plot(max_index, max_value, marker='o', linestyle='-', color='r', label="Line Plot")
    # Labels & Title
    plt.xlabel("X-Axis")
    plt.ylabel("Y-Axis")
    plt.title("Simple Line Plot")
    plt.legend()  # Show legend
    plt.show()

         
    return scan_data


sample_scan_polar = np.array([
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

# takes cartizian data and turns it polar:
# sample_scan_polar = np.zeros((len(sample_scan_cart),2))
# for i in range(len(sample_scan_cart)):

#     sample_scan_polar[i] = loc_to_rangeangle(p,sample_scan_cart[i]).ravel()

# print("sample_scan_polar",sample_scan_polar)
# print("sample_scan_cart",sample_scan_cart)

#r, theta, __ = find_corner(sample_scan_cart)

a = find_cuner(sample_scan_polar,p)

# x_sin = np.linspace(0, 4*np.pi, 100)
# y_sin = 2*np.sin(x_sin) + x_sin

#scan_maxima = argrelextrema(sample_scan_polar[:,0], np.greater)[0]

#print("scan_maxima:",scan_maxima)

#plt.plot(sample_scan_cart[:,1], sample_scan_polar[:,0], marker='o', linestyle='-', color='b', label="Line Plot")
#plt.plot(sample_scan_cart[:,0], sample_scan_cart[:,1], marker='o', linestyle='-', color='b', label="Line Plot")
#plt.plot(x_sin[sin_minima], y_sin[sin_minima], marker='o', linestyle='-', color='r', label="Line Plot")
#plt.plot(infection_point_cart[0], infection_point_cart[1], marker='o', linestyle='-', color='r', label="Line Plot")
#plt.plot(bot_pose_cart[0], bot_pose_cart[1], marker='o', linestyle='-', color='r', label="Line Plot")
# Labels & Title
# plt.xlabel("X-Axis")
# plt.ylabel("Y-Axis")
# plt.title("Simple Line Plot")
# plt.legend()  # Show legend
# plt.show()