
import numpy as np


r = [1,2,3,4,5]
theta = [-1,-0.5,0,0.5,1]

x = r * np.cos(theta)
y = r * np.sin(theta) 
print(x)
print(y)