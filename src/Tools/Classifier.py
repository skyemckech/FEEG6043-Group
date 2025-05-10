from import_functions import format_scan, combine_scans

class Classifier:
    def __init__(self):
        filepath = "src/Tools/training_data/"
        corner_no_noise = format_scan(filepath+"corner_0d_0mm", 'corner')
        corner_low_noise = format_scan(filepath+"corner_1d_5mm.json", 'corner')
        corner_high_noise = format_scan(filepath+"corner_3d_15mm.json", 'corner')

        wall_no_noise = format_scan(filepath+"wall_0d_0mm.json", 'wall')
        wall_low_noise = format_scan(filepath+"wall_1d_5mm.json", 'wall')
        wall_high_noise = format_scan(filepath+"wall_3d_15mm.json", 'wall')

        object_no_noise = format_scan(filepath+"circle_0d_0mm.json", 'object')
        object_low_noise = format_scan(filepath+"circle_1d_5mm.json", 'object')
        object_high_noise = format_scan(filepath+"circle_3d_15mm.json", 'object')

        self.no_noise = combine_scans(corner_no_noise,wall_no_noise,object_no_noise)
        self.low_noise = combine_scans(self.no_noise, corner_low_noise,wall_low_noise,object_low_noise)
        self.high_noise = combine_scans(self.low_noise, corner_high_noise,wall_high_noise,object_high_noise)

    def classify(self, label):
        