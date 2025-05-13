from .import_functions import format_scan, combine_scans, find_thetas
import numpy as np

class Classifier:
    def __init__(self):
        self.data = None
        self.classifier = None

        filepath = "src/Tools/training_data/"
        corner_no_noise = format_scan(filepath+"corner_0.json", 'corner')
        corner_sider = format_scan(filepath+"corner_0_sider.json", 'corner')
        corner_sidel = format_scan(filepath+"corner_0_sidel.json", 'corner')
        # corner_low_noise = format_scan(filepath+"corner_1d_5mm.json", 'corner')
        # corner_high_noise = format_scan(filepath+"corner_3d_15mm.json", 'corner')

        wall_no_noise = format_scan(filepath+"wall_0.json", 'wall')
        wall_wides = format_scan(filepath+'wall_wides.json', 'wall')
        # wall_low_noise = format_scan(filepath+"wall_1d_5mm.json", 'wall')
        # wall_high_noise = format_scan(filepath+"wall_3d_15mm.json", 'wall')

        object_no_noise = format_scan(filepath+"circle_0.json", 'object')
        # object_low_noise = format_scan(filepath+"circle_1d_5mm.json", 'object')
        # object_high_noise = format_scan(filepath+"circle_3d_15mm.json", 'object')

        nothing = format_scan(filepath+"nothing.json",'None')

        no_noise = combine_scans(corner_no_noise,corner_sider,corner_sidel, wall_no_noise,wall_wides, object_no_noise, nothing)
        # low_noise = combine_scans(no_noise, corner_low_noise,wall_low_noise,object_low_noise)
        # high_noise = combine_scans(low_noise, corner_high_noise,wall_high_noise,object_high_noise)

        self.data = no_noise
        # self.data = [no_noise,low_noise,high_noise]

    def train_classifier(self, label=str):
        data = self.data
        for i in range(len(data)):
            if data[i].label is not label:
                data[i].label = 'not '+ label
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

