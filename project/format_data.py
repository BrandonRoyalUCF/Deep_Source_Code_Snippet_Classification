import random
import numpy as np
from copy import deepcopy

# Seed randoms here so we get the same results if repeated
np.random.seed(7)
random.seed(7)


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

