

class DataPoint(object):

    def __init__(self, features, label):

        self.features = features
        self.label = label

class Dataset(object):

    def __init__(self, data_points):

        self.all_data_points = data_points

class Experiment(object):

    def __init__(self):

        self.ten_fold_divisions = []
        