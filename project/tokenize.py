# This file contains the main tokenization methods.
# Methods are declared at the top and the main execution is at the bottom
# Changes to main are needed depending on what the task to be done is.
# File paths will need to be updated based on who is running the code.

import os
import pickle 

#########################################
# Functions Used By Other Functions Here
# ********KEEP THESE AT THE TOP********
#########################################

def is_alpha(ch):
    """Checks if a character is in ['a' - 'z'] or ['A' - 'Z']
    """
    #if lowercase alpha
    if(ch >= 'a' and ch <= 'z'):
        return True
    if(ch >= 'A' and ch <= 'Z'):
        return True
    return False

def is_numeric(ch):
    """Checks if a character is in ['0' - '9']
    """
    if(ch >= '0' and ch <= '9'):
        return True
    return False

def is_a_number(string):
    """Checks if a string is a number
    """

    for ch in string:
        if(not is_numeric(ch)):
            return False
    return True

#########################################
# Functions Called By Code In Task Block
# *****KEEP THESE ABOVE MAIN BLOCK*****
#########################################

def get_tokens_and_counts(dataset_path):
    """
    Parses all files in the dataset to extract tokens and their counts across the dataset.

    For now we have 3 classes: alphabet token, numeric token, and non alpha-numeric.
    For alphabet tokens we build a token string until a non alpha character is found.
    For numeric tokens we build a token string until a non numeric character is found.
    For non alpha-numeric all tokens are single characters.

    Args:
        dataset_path: A string containing the base folder of the dataset.
    
    Returns:
        A dictionary mapping each token (key) to the token's count (value)
        across the whole dataset.
    
    """

    #get the file path for every file in the dataset
    all_files = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if(file.endswith('.txt')):
                all_files.append(os.path.join(root, file))

    #tokens will be our dictionary mapping the token (key) to its count (value)
    tokens = {}
    count=0

    #for every file in the dataset, parse the file one character at a time.
    #if the character is non alpha numeric, treat it as an individual token.
    #if the character is in ['a' - 'z'] then build a word token until we find a non alphabet character
    #if the character is in ['0' - '9'] then build a number token until we find a non numerical character
    #note: we always set the text to lower case for simplicity (may be bad? change later?)
    for fil in all_files:
        print('Starting ' + str(count))
        count+=1
        text = open(fil, 'r')

        #for each line in the file
        for line in text:
            line = line.lower()

            #holds the current token if we are building a number or a word
            cur_token = ''
            for i in range(len(line)):

                #get the current character 
                ch = line[i]

                #if the current character is an alphabet character
                if(is_alpha(ch)):

                    #if the cur token is not empty
                    if(cur_token != ''):

                        #if we have been building a number token, increment that token's count
                        if(is_numeric(cur_token[0])):
                            if(cur_token in tokens.keys()):
                                tokens[cur_token] += 1
                            else:
                                tokens[cur_token] = 1

                            #set the cur token to the current character
                            cur_token = ch

                        #if we have been building a alphabetic token just add the current character onto it
                        elif(is_alpha(cur_token[0])):
                            cur_token += ch

                        else:
                            #we shouldnt get here, if we do raise an error
                            #since all non alpha numeric characters are treated as single character tokens
                            raise Exception('Invalid token')

                    else:
                        #if the current token is empty, start building an alphabetic token
                        cur_token += ch

                #if the current character is numeric
                elif(is_numeric(ch)):

                    #if the cur token is not empty
                    if(cur_token != ''):

                        #if we have been building a letter token, increment that token's count
                        if(is_alpha(cur_token[0])):
                            if(cur_token in tokens.keys()):
                                tokens[cur_token] += 1
                            else:
                                tokens[cur_token] = 1

                            #set the cur token to the current character
                            cur_token = ch

                        #if we have been building a numeric token just add the current character onto it
                        elif(is_numeric(cur_token[0])):
                            cur_token += ch

                        else:
                            #we shouldnt get here, if we do raise an error
                            #since all non alpha numeric characters are treated as single character tokens
                            raise Exception('Invalid token')

                    else:
                        #if the current token is empty, start building a number token
                        cur_token += ch
                            
                    
                #if the character is not alpha numeric
                else:

                    #if our cur token is not empty then a alpha or numeric token was being built
                    # increment its count and reset cur_token
                    if(cur_token != ''):
                        if(cur_token in tokens.keys()):
                            tokens[cur_token] += 1
                        else:
                            tokens[cur_token] = 1
                        cur_token = ''
                    
                    #increment the count of the non alpha numeric token
                    #for now we always treate this as a single character
                    #FUTURE: We could pre hardcode all ops like '++', '+=', ect. so we have multi character tokens here
                    if(ch in tokens.keys()):
                        tokens[ch] += 1
                    else:
                        tokens[ch] = 1

            #make sure we add the last token if needed
            if(cur_token != ''):
                if(cur_token in tokens.keys()):
                    tokens[cur_token] += 1
                else:
                    tokens[cur_token] = 1
                cur_token = ''

    
    #return the token dictionary
    #Key = the token as a string, val = the count the token was found in the dataset
    return tokens

def save_obj_pickle(obj, output_path):
    """Saves an object as a pickle file to the specified path
    
    Args:
        obj: a Python object
        output_path: the full path, including the file name, where the pickle file will be saved
    
    """
    with open(output_path, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_object_pickle(object_path):
    """Loads an object as a pickle file.

    Args:
        file_path: the full path, including the file name, where the pickle file is stored
    
    Returns:
        The object that was serialized in the picle object.
    """

    with open(object_path, 'rb') as handle:
        obj = pickle.load(handle)

    return obj

def analyze_token_counts(token_count_dict):
    """Prints the min, max, mean, median, and every 10% count distribution data.
    
    Args:
        token_count_dict: a dictionary with the token as the key and its count as the value
    """


    #add all counts to a list 
    all_counts = []
    for key in token_count_dict.keys():
        all_counts.append(token_count_dict[key])

    #sort the counts
    all_counts.sort()

    #number of counts (= number of tokens)
    num_counts = len(all_counts)

    #max and min value
    max_count = all_counts[num_counts-1]
    min_count = all_counts[0]

    #mean value
    mean_count = sum(all_counts) / num_counts

    #median value
    if(num_counts % 2 == 0):
        median1 = all_counts[num_counts//2]
        median2 = all_counts[num_counts//2 - 1]
        median = (median1 + median2)/2
    else:
        median = all_counts[num_counts//2]

    print('Total of ' + str(num_counts) + ' tokens in the dataset.')
    print('Min count value = ' + str(min_count))
    print('Max count value = ' + str(max_count))
    print('Mean count value = ' + str(mean_count))
    print('Median count value = ' + str(median))
    print()

    #next prints out distribution data about the counts
    ten_percent = int(num_counts * .1)
    cur_index = ten_percent
    counter = 90
    while(cur_index < num_counts):
        print(str(counter) + '%% of the tokens have a count >= ' + str(all_counts[cur_index]))
        counter-=10
        cur_index += ten_percent
    
def remove_numerical_tokens(token_dict):
    """Deletes all numerical tokens from the dictionary

    Args:
        token_dict: token dictionary where the key is the token. (value does not matter here)

    Returns:
        Returns the token dictionary with the numerical tokens removed and '<num>' added if it was not present before.
    """

    #first add the '<nums>' token in the dictionary if it is not present
    num_token = '<nums>'
    if(num_token not in token_dict.keys()):
        token_dict[num_token] = 0

    token_keys_to_delete = []

    #search over every token, if it is a number then delete it and increase the '<nums>' token count
    for key in token_dict.keys():

        if(is_a_number(key)):
            token_keys_to_delete.append(key)
            token_dict[num_token] +=1

    for key in token_keys_to_delete:
        del token_dict[key]

    return token_dict

def remove_tokens(token_dict, list_tokens_to_remove):
    """Remove all tokens in the given list if the token is present in the dictionary as the key value.
    If the token is not present, nothing is done.
    
    Args:
        token_dict: a python dictionary where the key is a token string and val is the token count
        list_tokens_to_remove: a list of strings that are tokens to be removed from the dictionary
    
    Returns:
        Returns a token dictionary with the specified tokens removed
    """

    # add the '<unknown>' token to the dict if not present
    if('<unknown>' not in token_dict.keys()):
        token_dict['<unknown>'] = 0

    token_keys_to_delete = []

    #for each token in the list, remove it from the dictionary if it is present
    for token in list_tokens_to_remove:
        if(token in token_dict.keys()):
            token_keys_to_delete.append(token)
            token_dict['<unknown>'] += 1

    for key in token_keys_to_delete:
        del token_dict[key]

    return token_dict

def remove_tokens_below_count(token_dict, min_count):

    # add the '<unknown>' token to the dict if not present
    if('<unknown>' not in token_dict.keys()):
        token_dict['<unknown>'] = 0

    token_keys_to_delete = []

    #for all tokens
    for key in token_dict.keys():

        #get the token's count
        count = token_dict[key]
        
        #if the count is below the min count, remove that token
        if(count < min_count):
            token_keys_to_delete.append(key)
            token_dict['<unknown>'] += 1

    for key in token_keys_to_delete:
        del token_dict[key]

    return token_dict

def remove_tokens_not_in_top_percent(token_dict, top_x_percent):

    # add the '<unknown>' token to the dict if not present
    if('<unknown>' not in token_dict.keys()):
        token_dict['<unknown>'] = 0

    #get all of the unique counts
    unique_token_counts = set()
    for key in token_dict.keys():
        unique_token_counts.add(token_dict[key])

    #move the counts into a sorted list
    list_unique_counts = [val for val in unique_token_counts].sort()

    #get how many unique counts there are
    num_counts = len(list_unique_counts)

    #change the whole percent into a decimal
    multiplier = top_x_percent / 100

    #index of count that is the min count to be in top x %
    starting_index = (num_counts - (num_counts  *  multiplier) - 1)

    #the min_count needed to keep the token
    min_count = list_unique_counts[starting_index]

    token_keys_to_delete = []
    
    #for all tokens delete the token if its below the min_count
    for key in token_dict.keys():

        #get the token's count
        count = token_dict[key]
        
        #if the count is below the min count, remove that token
        if(count < min_count):
            token_keys_to_delete.append(key)
            token_dict['<unknown>'] += 1

    for key in token_keys_to_delete:
        del token_dict[key]

    return token_dict

def create_token_id_dict(token_dict):

    from collections import OrderedDict

    all_tokens = []
    for token in token_dict.keys():
        all_tokens.append(token)

    all_tokens.sort()

    token_id_dict = OrderedDict()
    id_num = 0
    for token in all_tokens:
        token_id_dict[token] = id_num
        id_num+=1

    return token_id_dict

def generate_token_id_python(token_id_dict, token_id_python_path):


    dict_text = open(token_id_python_path, 'w+')
    dict_text.write('tokens = {\n')
    
    for key, val in token_id_dict.items():
        dict_text.write('\t\"' + key + '\" : ' + str(val) + ',\n')
        
    dict_text.write('\t}\n')

    dict_text.close()

def get_token_vector_for_all_files(dataset_path, token_id_dict):

    #define these for later use
    num_token_string = '<nums>'
    unkown_token_string = '<unknown>'

    #get the file path for every file in the dataset
    all_files = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if(file.endswith('.txt')):
                all_files.append(os.path.join(root, file))

    #token_vectors will be a dictionary mapping the file name with folder name (key) to its list of token values (value)
    #an example would be: key = '/javascript/1789436.txt', value = [45, 700, 2, 11, ... , 77]
    token_vectors = {}

    count = 0

    #for every file in the dataset, parse the file one character at a time.
    #if the character is non alpha numeric, treat it as an individual token.
    #if the character is in ['a' - 'z'] then build a word token until we find a non alphabet character
    #if the character is in ['0' - '9'] then build a number token until we find a non numerical character
    #note: we always set the text to lower case for simplicity (may be bad? change later?)
    for fil in all_files:
        print('Starting ' + str(count))
        count+=1
        text = open(fil, 'r')

        token_vec = []

        #for each line in the file
        for line in text:
            line = line.lower()

            #holds the current token if we are building a number or a word
            cur_token = ''
            for i in range(len(line)):

                #get the current character 
                ch = line[i]

                #if the current character is an alphabet character
                if(is_alpha(ch)):

                    #if the cur token is not empty
                    if(cur_token != ''):

                        #if we have been building a number token, add '<num>' token to the vector
                        if(is_numeric(cur_token[0])):
                            token_id = token_id_dict[num_token_string]
                            token_vec.append(token_id)
                            
                            #set the cur token to the current character
                            cur_token = ch

                        #if we have been building a alphabetic token just add the current character onto it
                        elif(is_alpha(cur_token[0])):
                            cur_token += ch

                        else:
                            #we shouldnt get here, if we do raise an error
                            #since all non alpha numeric characters are treated as single character tokens
                            raise Exception('Invalid token')

                    else:
                        #if the current token is empty, start building an alphabetic token
                        cur_token += ch

                #if the current character is numeric
                elif(is_numeric(ch)):

                    #if the cur token is not empty
                    if(cur_token != ''):

                        #if we have been building a letter token, increment that token's count
                        if(is_alpha(cur_token[0])):
                            if(cur_token in token_id_dict.keys()):
                                token_id = token_id_dict[cur_token]
                            else:
                                token_id = token_id_dict[unkown_token_string]

                            token_vec.append(token_id)

                            #set the cur token to the current character
                            cur_token = ch

                        #if we have been building a numeric token just add the current character onto it
                        elif(is_numeric(cur_token[0])):
                            cur_token += ch

                        else:
                            #we shouldnt get here, if we do raise an error
                            #since all non alpha numeric characters are treated as single character tokens
                            raise Exception('Invalid token')

                    else:
                        #if the current token is empty, start building a number token
                        cur_token += ch
                            
                    
                #if the character is not alpha numeric
                else:

                    #if our cur token is not empty then a alpha or numeric token was being built
                    # increment its count and reset cur_token
                    if(cur_token != ''):

                        #if we have been building a letter token, increment that token's count
                        if(is_alpha(cur_token[0])):
                            if(cur_token in token_id_dict.keys()):
                                token_id = token_id_dict[cur_token]
                            else:
                                token_id = token_id_dict[unkown_token_string]

                            token_vec.append(token_id)


                        #if we have been building a numeric token just add the current character onto it
                        elif(is_numeric(cur_token[0])):
                            token_id = token_id_dict[num_token_string]
                            token_vec.append(token_id)

                        else:
                            #we shouldnt get here, if we do raise an error
                            #since all non alpha numeric characters are treated as single character tokens
                            raise Exception('Invalid token')
                        
                        cur_token = ''
                    
                    #increment the count of the non alpha numeric token
                    #for now we always treate this as a single character
                    #FUTURE: We could pre hardcode all ops like '++', '+=', ect. so we have multi character tokens here
                    if(ch in token_id_dict.keys()):
                            token_id = token_id_dict[ch]
                    else:
                        token_id = token_id_dict[unkown_token_string]

            #make sure we add the last token if needed
            if(cur_token != ''):
                #if we have been building a letter token, increment that token's count
                if(is_alpha(cur_token[0])):
                    if(cur_token in token_id_dict.keys()):
                        token_id = token_id_dict[cur_token]
                    else:
                        token_id = token_id_dict[unkown_token_string]

                    token_vec.append(token_id)


                #if we have been building a numeric token just add the current character onto it
                elif(is_numeric(cur_token[0])):
                    token_id = token_id_dict[num_token_string]
                    token_vec.append(token_id)

        #get the file name and the one folder above it
        #NOTE: This is pretty hard coded for linux
        file_name_only = os.path.basename(fil)
        parts = fil.split('/')

        #folder is the programming language name 
        folder = parts[len(parts)-2]

        file_name_with_folder = folder + '/' + file_name_only

        #add the token vector for the file into the dictionary
        token_vectors[file_name_with_folder] = token_vec
    
    #return the token dictionary
    #Key = the token as a string, val = the count the token was found in the dataset
    return token_vectors




##################################
# *********Task Block************
##################################

# Note: This block has methods for specific tasks useful for us
# Keep these methods below the above two blocks.
# Sorry for the annoyingly long names but I wanted them to be specific.

######################################################################
# Task: Parse Original Dataset and Collect All Tokens and Their Counts
#######################################################################
def task_parse_dataset_get_counts(dataset_basepath):
    """Task: Parses the whole dataset to get each unique token and its count.
    Three types of tokens: 
        1) Alphabet Token (ex: 'hello')
        2) Numeric Token (ex: '12345')
        3) Non Alpha-Numeric Token (ex: '+', '\t', ' ')
    
    This task will parse the whole dataset, create a dictionary where the 
    key is the token value as a string and the value is the count of that token,
    save the dictionary as a pickle file to the current directory, and display
    some stats about the token counts.
    
    Args:
        dataset_basepath: base filepath of the dataset directory

    Returns:
        A dictionary of all tokens and their counts. 
    """

  

    # retrieve all tokens in the dataset and their counts
    token_count_dict = get_tokens_and_counts(dataset_basepath)

    # path for dictionary to be saved, currently saved as a pickle file
    token_count_dict_path = os.getcwd() + '/all_token_count_dict.pkl'

    # save the token_count_dict to a pickle file
    save_obj_pickle(token_count_dict, token_count_dict_path)

    # display some statistics about the token counts for the dataset
    analyze_token_counts(token_count_dict)

    # return the count dict
    return token_count_dict

################################################################################
# Task: Clean Token Dictionary and Save a Token to Token ID Dictionary to a File 
################################################################################

def task_clean_token_dict_create_token_ids(all_token_dict, min_count=-1, top_x_percent=-1, list_invalid_tokens=None):
    """Task: Remove tokens we do not care about and autogenerate python code to create
    the token to token id code. Also save token to id dict as a pickle file.

    Token Cleaning:
        1) Remove all numerical tokens, create a single <num> token
        2) If list_invalid_tokens is set, remove those specific tokens
        3) If min_count is set, make all tokens below that count to be <unknown> tokens.
        4) If top_x_percent_count is set, make all tokens not in the top x% (by count) <unknown> tokens 

    Notes: 
        1) top_x_percent is used over min_count if both are set
        2) min_count and top_x_percent_count are considered AFTER removing numerical tokens and list_invalid_tokens.

    Args:
        all_token_dict: a dictionary holding the token to token counts
        min_count: the min count needed for a token to be used
        top_x_percent: integer representing the top % (by count) of the tokens that should be used
        list_invalid_tokens: specific tokens to be removed

    Returns:
        A dictionary of token to token id. This dictionary is also saved as a pickle file.
        The tokens are sorted here by string value using python's build in sort 
        so that doing this twice will keep the data token to id values the same (unless params are different).
    """
    
    token_dict = all_token_dict

    # add the '<unknown>' token to the dict if not present
    if('<unknown>' not in token_dict.keys()):
        token_dict['<unknown>'] = 0

    # remove all numerical tokens from the dictionary and add the '<num>' token
    token_dict = remove_numerical_tokens(token_dict)

    # remove all tokens in list_invalid_tokens
    if(list_invalid_tokens is not None):
        token_dict = remove_tokens(token_dict, list_invalid_tokens)

    # remove tokens based on top_x_percent or min_count
    # if both are not -1, only top_x_perent is used
    if(top_x_percent != -1):
        token_dict = remove_tokens_not_in_top_percent(token_dict, top_x_percent)
    elif(min_count != -1):
        token_dict = remove_tokens_below_count(token_dict, min_count)

    # create a token to token id dictionary, the tokens will be sorted first
    token_id_dict = create_token_id_dict(token_dict)

    # path for token to token id dictionary to be saved, currently saved as a pickle file
    token_id_dict_path = os.getcwd() + '/token_id_dict.pkl'

    # save the token_id_dict to a pickle file
    save_obj_pickle(token_id_dict, token_id_dict_path)

    # path for the python code generated for the token dict
    token_id_python_path = os.getcwd() + '/token_id_dict.py'

    # generate python code for the token_id_dict
    generate_token_id_python(token_id_dict, token_id_python_path)

    # return the token to token id dictionary
    return token_id_dict

def task_create_token_vectors_for_all_files(dataset_basepath, token_id_dict):

    # get a dictionary mapping each file name with the programming folder included (key)
    # to the token vector, a list of token ids in the order they were found (value)
    token_vectors_dict = get_token_vector_for_all_files(dataset_basepath, token_id_dict)

    # path for token to token id dictionary to be saved, currently saved as a pickle file
    token_vectors_dict_path = os.getcwd() + '/token_vectors_dict.pkl'

    # save the token_vectors_dict to a pickle file
    save_obj_pickle(token_vectors_dict, token_vectors_dict_path)

    return token_vectors_dict


##################################
# *********Main Block************
##################################

# This is the main entry block where tasks or methods can be called.
# Tasks are just predefined tasks I know we will need to do but a single
# method can also be called if needed.

#NOTE: Warning task_create_token_vectors_for_all_files only works on linux because of path stuff

# Dataset Path for Brandon's Lab Desktop
dataset_base_path = '/home/brandon/NLP_Project/datasets'

# token_count_dict = task_parse_dataset_get_counts(dataset_base_path)
# print('Done with creating token count dict.')
token_count_dict_path = os.getcwd() + '/all_token_count_dict.pkl'
token_count_dict = load_object_pickle(token_count_dict_path)

# token_id_dict = task_clean_token_dict_create_token_ids(token_count_dict, min_count=10)
# print('Done with creating token id dict.')
token_id_dict_path = os.getcwd() + '/token_id_dict.pkl'
token_id_dict = load_object_pickle(token_id_dict_path)

# token_vectors_dict = task_create_token_vectors_for_all_files(dataset_base_path, token_id_dict)
# print('Done with creating the token vector dict.')
token_vectors_dict_path = os.getcwd() + '/token_vectors_dict.pkl'
token_vectors_dict = load_object_pickle(token_vectors_dict_path)

print('Done with all tasks.')









