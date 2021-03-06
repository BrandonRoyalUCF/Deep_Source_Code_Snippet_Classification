import random
import numpy as np
from copy import deepcopy
import os
from keras.preprocessing.sequence import pad_sequences

# our own files
from tokenizer import load_object_pickle
from tokenizer import save_object_pickle
from dataset import Dataset
from dataset import DataPoint

# Seed randoms here so we get the same results if repeated
np.random.seed(7)
random.seed(7)


def get_max_token_value(data_dict):
    """Finds the max token value within all data samples
    """
    max_val = -1

    for key in data_dict.keys():
        for val in data_dict[key]:
            if(val > max_val):
                max_val = val

    return max_val

def unison_shuffled_copies(a, b):
    """Shuffles two lists in unison. 
    Allows a list of features and a list of labels to be shuffled the same way.
    """
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def get_label_from_name(programming_language):
    """Returns the integer label of the given programming language.

    Args:
        programming_language: String representation of programming language.

    Returns:
        The label, an integer 1-21, representing the given programming language.
    """
    programming_language = programming_language.lower()

    pl_to_label = {  'bash' : 1, 'c' : 2, 'c#' : 3, 'c++' : 4, 
                     'css' : 5, 'haskell' : 6, 'html' : 7, 'java' : 8, 
                     'javascript' : 9, 'lua' : 10, 'markdown' : 11, 
                     'objective-c' : 12, 'perl' : 13, 'php' : 14, 'python' : 15, 
                     'r' : 16, 'ruby' : 17, 'scala' : 18, 'sql' : 19, 'swift' : 20, 
                     'vb.net' : 21
                  }
    
    if(programming_language not in pl_to_label.keys()):
        raise Exception('Warning invalid programming language key.')
    
    return pl_to_label[programming_language]

def ten_fold_split(data_points):
    """Shuffles the data and splits the dataset into k equal divisions.
    
    Args:
        data_points: A list of DataPoint objects.

    Returns:
        A list of ten lists. Each list contains 10% of the dataset (still DataPoint objects)
    """

    # shuffle the input array
    random.shuffle(data_points)

    divisions = []

    ten_percent = int(len(data_points) * .1)
    count = 1
    curlist = []

    for point in data_points:
        if(count == ten_percent):
            divisions.append(deepcopy(curlist))
            curlist = []
            count = 1
        
        curlist.append(point)
        count+=1

    if(len(curlist) > 0):
        cur_i = 0
        for point in curlist:
            divisions[cur_i].append(point)
            cur_i+=1
            if(cur_i > 9):
                cur_i = 0

    return divisions

def create_dataset(data_dict):
    """Creates a Dataset object from the data in dictionary format
    
    Args:
        data_dict: A dictionary mapping where the key is the file name and
            language folder (ex: 'python/123432.txt') and the value is the 
            tokenized feature vector for that file.

    Returns:
        A Dataset object with all data samples inside. The Dataset object has
        a list of DataPoint objects which each have a 'label' (integer label)
        and 'features' (the token vector)
    """
    return True

def pad_and_truncate(dataset_path, set_length):
    """ Pads / Truncates feature vectors from the dataset dictionary to length set_length and saves as pkl dataset obj

    Note: Uses 0.0 value for padding

    Args:
        dataset_path: A file path to the dataset dictionary pkl file
        set_length: The specified length to pad / truncate

    Returns:
        A two dimensional feature vector that has been padded / truncated to length set_length
    """

    # load the dictionary from the pickle file
    dataset = load_object_pickle(dataset_path)

    # create feature vector list
    feature_vectors = []

    # create feature vector label list
    feature_vectors_labels = []

    # populate feature vector/label lists, then change type to np array for broadcasting
    for sample in dataset.all_data_points:
        feature_vectors.append(sample.features)
        feature_vectors_labels.append(sample.label)
    feature_vectors = np.array(feature_vectors)
    feature_vectors_labels = np.array(feature_vectors_labels)

    # truncate / pad to pre-determined length (see feature_vector_length_hist.png)
    feature_vectors = pad_sequences(feature_vectors, maxlen=set_length, truncating='post')

    # store the truncated / padded feature vectors to the dictionary dataset
    i = 0
    for sample in dataset.all_data_points:
        sample.features = feature_vectors[i]
        i += 1

    # save to dictionary using pickle
    save_object_pickle(dataset, os.getcwd() + '/dataset_pad_and_trunc.pkl')

    # save to npy file
    np.save('train_data.npy', feature_vectors)
    np.save('train_data_labels.npy', feature_vectors_labels)

    return


##################################
# *********Task Block************
##################################

# Note: This block has methods for specific tasks useful for us
# Keep these methods below the above two blocks.
# Sorry for the annoyingly long names but I wanted them to be specific.

######################################################################
# Task: Create Dataset Object From Token Vectors Dictionary
#######################################################################

def task_create_dataset_from_token_dictionary(token_vectors_dict_path):

    # load the dictionary from the pickle file
    token_vectors_dict = load_object_pickle(token_vectors_dict_path)

    # determine the current max token value
    max_val = get_max_token_value(token_vectors_dict)

    # for each sample in the dataset, create a Datapoint object (contains feats and label)
    all_samples = []
    for key in token_vectors_dict.keys():
        label = get_label_from_name(key.split('/')[0])
        features = token_vectors_dict[key]

        sample = DataPoint(features, label)
        all_samples.append(sample)

    dataset = Dataset(all_samples, max_val)
    return dataset

def task_pad_truncate_data_and_save_new_datset_object(dataset_path, set_length):
    pass
    # load the dictionary from the pickle file
    #token_vectors_dict = load_object_pickle(dataset_path)

    # pad or truncate all feature vectors in the dataset to the given length



##################################
# *********Main Block************
##################################

# create a Dataset object from the token vector dictionary
# token_vectors_dict_path = os.getcwd() + '/token_vectors_dict.pkl'
# dataset = task_create_dataset_from_token_dictionary(token_vectors_dict_path)
# save_object_pickle(dataset, os.getcwd() + '/dataset.pkl')

if __name__ == '__main__':
    pad_and_truncate(os.getcwd() + '/dataset.pkl', 200)

