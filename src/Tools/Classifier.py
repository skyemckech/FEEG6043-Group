from .import_functions import format_scan, combine_scans, find_thetas
import numpy as np

class Classifier:
    def __init__(self):
        self.data = None
        self.classifier = None

        filepath = "src/Tools/training_data/"
        gameroajgood = [0,0,0,0,0,1,1,1,1,1,1]
        corner_no_noise = format_scan(filepath+"NOW.json", gameroajgood)

        # corner_no_noise = format_scan(filepath+"corner_0.json", 'corner')
        # corner_sider = format_scan(filepath+"corner_0_sider.json", 'corner')
        # corner_sidel = format_scan(filepath+"corner_0_sidel.json", 'corner')

        # wall_no_noise = format_scan(filepath+"wall_0.json", 'wall')
        # wall_wides = format_scan(filepath+'wall_wides.json', 'wall')

        # object_no_noise = format_scan(filepath+"circle_0.json", 'object')


        # nothing = format_scan(filepath+"nothing.json",'None')

        # no_noise = combine_scans(corner_no_noise,corner_sider,corner_sidel, wall_no_noise,wall_wides, object_no_noise)

        self.data = corner_no_noise
        # self.data = [no_noise,low_noise,high_noise]

    def train_classifier(self):
        gjaisfagsos = [0,0,0,0,0,1,1,1,1,1,1]
        data = self.data[:120,:]
        for i in range(len(data)):
            if gjaisfagsos[i] == 0:
                data[i].label = 'not corner'
            else:
                data[i].label = 'corner'
        theta1, theta2, self.classifier, _,_ = find_thetas(data)
        print("theta 1 is:",theta1, "theta 2 is:",theta2)

    def check_classifier(self):
        test_d = format_scan("src/Tools/training_data/realsquare3x.json",None)
        return test_d


# cornerClassifier = Classifier()
# cornerClassifier.train_classifier('corner')
# observation = cornerClassifier.data[5]
# proba = cornerClassifier.classifier.predict_proba([observation.data_filled[:,0]])
# label = (cornerClassifier.classifier.classes_[np.argmax(proba)])
# print(proba)
# print(label)

