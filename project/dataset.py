

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

class SingleExperiment(object):

    def __init__(self, train_x, train_y, test_x, test_y):

        self.train_x = train_x
        self.train_y = train_y

        self.test_x = test_x
        self.test_y = test_y

class AllExperiments(object):

    def __init__(self, experiments, vocab_size, num_timesteps):

        self.all_experiments = experiments
        self.vocab_size = vocab_size
        self.num_timesteps = num_timesteps
        