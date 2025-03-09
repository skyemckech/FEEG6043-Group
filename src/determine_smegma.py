import numpy as np

wheel_diameter = 0.074
wheel_separation = 0.162

smegma_wrl = 0.02
smegma_diameter = 0.02*wheel_diameter
smegma_separation = 0.1*wheel_separation

first_term = (wheel_diameter/2/np.sqrt(2)*smegma_wrl)
print(first_term)

second_term = smegma_diameter/wheel_diameter
print(second_term)