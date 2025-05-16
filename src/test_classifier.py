from Tools import *
from Libraries import *
from matplotlib import pyplot as plt

cornerClassifier = Classifier()
cornerClassifier.train_classifier()
testd = cornerClassifier.check_classifier()

# observation = cornerClassifier.data[52]
observation = testd[:20]


# [0,1,2,3,4,5,6,7,8]
[0,0,0,0,0,0]
# ['not corner','not corner','not corner','not corner','not corner', 'corner, 'corner', 'corner']

p = Vector(3); 

p[0] = 0 #Northings
p[1] = 0 #Eastings
p[2] = 0 #Heading (rad)

x_bl = 0; y_bl = 0
lidar = RangeAngleKinematics(x_bl, y_bl, distance_range = [0.1, 1], scan_fov = np.deg2rad(60), n_beams = 30)
for obs in observation:
    
    obs.data_filled = obs.data_filled[:120,:]
    obs.data = obs.data[:120,:]
    proba = cornerClassifier.classifier.predict_proba([obs.data_filled[:,0]])
    print(proba)
    obs.label = (cornerClassifier.classifier.classes_[np.argmax(proba)])
    show_scan(p, lidar, obs.data)

    obs.data_filled = obs.data_filled[obs.data[:, 0] > 0.02]
    if obs.label == 'corner' and len(obs.data_filled) > 10:
        z_lm = Vector(2)
        z_lm[0], z_lm[1], loc = find_corner(obs, 0.003) # we can set a lower threshold here

        if loc is not None:  
            # plot the raw sensor readings
            plt.plot(np.rad2deg(obs.data_filled[:, 1]), obs.data_filled[:, 0], 'g.', label='Observations')
            plt.plot(np.rad2deg(obs.data_filled[loc, 1]), obs.data_filled[loc, 0], 'ro', label='Observations')
            plt.xlabel('Bearing (sensor frame), degrees')
            plt.ylabel('Range, m')
            plt.show()
        
            obs.ne_representative=lidar.rangeangle_to_loc(p,z_lm)

