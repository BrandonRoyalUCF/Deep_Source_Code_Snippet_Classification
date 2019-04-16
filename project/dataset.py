

class DataPoint(object):

    def __init__(self, features, label):

        self.features = features
        self.label = label

class Dataset(object):

    def __init__(self, data_points, max_token_value):

        # a list of all DataPoint objects (samples) from the dataset
        self.all_data_points = data_points
        
        # since we will have to pad and truncate token vectors, give a specific token
        # value to the padding value
        self.pad_token_number = max_token_value + 1

        # vocab_size is the the highest token value (which is pad_token_number) + 1
        self.vocab_size = self.pad_token_number + 1

class Experiment(object):

    def __init__(self):

        self.ten_fold_divisions = []
        