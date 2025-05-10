from import_functions import format_scan, combine_scans, find_thetas

class Classifier:
    def __init__(self):
        filepath = "src/Tools/training_data/"
        corner_no_noise = format_scan(filepath+"corner_0d_0mm.json", 'corner')
        corner_low_noise = format_scan(filepath+"corner_1d_5mm.json", 'corner')
        corner_high_noise = format_scan(filepath+"corner_3d_15mm.json", 'corner')

        wall_no_noise = format_scan(filepath+"wall_0d_0mm.json", 'wall')
        wall_low_noise = format_scan(filepath+"wall_1d_5mm.json", 'wall')
        wall_high_noise = format_scan(filepath+"wall_3d_15mm.json", 'wall')

        object_no_noise = format_scan(filepath+"circle_0d_0mm.json", 'object')
        object_low_noise = format_scan(filepath+"circle_1d_5mm.json", 'object')
        object_high_noise = format_scan(filepath+"circle_3d_15mm.json", 'object')
    
        no_noise = combine_scans(corner_no_noise,wall_no_noise,object_no_noise)
        low_noise = combine_scans(no_noise, corner_low_noise,wall_low_noise,object_low_noise)
        high_noise = combine_scans(low_noise, corner_high_noise,wall_high_noise,object_high_noise)

        self.data = [no_noise,low_noise,high_noise]

    def classify(self, noise, label):
        data = self.data[noise]
        for i in range(len(data)):
            if data[i].label is not label:
                data[i].label = 'not '+ label

        theta1, theta2, _,_,_ = find_thetas(data)
        print("theta 1 is:",theta1, "theta 2 is:",theta2)

cornerClassifier = Classifier()
cornerClassifier.classify(0,'corner')