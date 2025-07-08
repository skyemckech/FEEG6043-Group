import numpy as np
from math_feeg6043 import Matrix

animal = [ 0.44607788, -0.78539816]
boat = [ 0.00446996, -0.78539816]

sigma_observe = Matrix(2,2)
sigma_observe[0,0] = 0.01 #1% of range
sigma_observe[0,1] = 0
sigma_observe[1,0] = np.deg2rad(1) #1 degree per metre range
sigma_observe[1,1] = 0

print (animal @ sigma_observe)