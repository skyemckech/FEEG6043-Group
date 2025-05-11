from Tools import *
from Libraries import *

cornerClassifier = Classifier()
cornerClassifier.train_classifier('corner')
testd = cornerClassifier.check_classifier()

# observation = cornerClassifier.data[52]
observation = testd[10]
proba = cornerClassifier.classifier.predict_proba([observation.data_filled[:,0]])
label = (cornerClassifier.classifier.classes_[np.argmax(proba)])
print(proba)
print(label)

p = Vector(3); 

p[0] = 0 #Northings
p[1] = 0 #Eastings
p[2] = 0 #Heading (rad)

x_bl = 0; y_bl = 0
lidar = RangeAngleKinematics(x_bl, y_bl, distance_range = [0.1, 1], scan_fov = np.deg2rad(60), n_beams = 30)

show_scan(p, lidar, observation.data)
	
