import os
import numpy as np

from dataset import Dataset
from dataset import DataPoint
from dataset import AllExperiments
from dataset import SingleExperiment

from model import create_simple_lstm_model
from model import create_conv_bidirect_lstm_model
from model import create_new_test_model
from model import save_model
from model import load_model

from tokenize_data import save_object_pickle
from tokenize_data import load_object_pickle

from keras.callbacks import ModelCheckpoint

from sklearn.metrics import confusion_matrix

def get_name_from_label(label):
    label_to_pl = {  1: 'bash', 2: 'c', 3: 'c#', 4: 'c++', 
                     5:'css', 6:'haskell', 7:'html', 8:'java', 
                     9:'javascript', 10:'lua', 11:'markdown', 
                     12:'objective-c', 13:'perl', 14:'php', 15:'python', 
                     16:'r', 17:'ruby', 18:'scala', 19:'sql', 20:'swift', 
                     21:'vb.net'
                  }
    return label_to_pl[label]

def get_class_count(labels):

    counts = {}

    for label in labels:
        if(label not in counts.keys()):
            counts[label] = 0
        
        counts[label] +=1

    for key in counts.keys():
        print('Class ' + str(key) + 'has ' + str(counts[key]))
    print()


        


def convert_labels_to_multiclass(all_labels, num_classes, num_timesteps):

    new_labels = []
    for label in all_labels:
        time_dist = []
        new_label = np.zeros(num_classes)
        new_label[label-1] = 1

        # uncomment below if doing Time-Distributed
        for i in range(int(num_timesteps/2)):
            time_dist.append(new_label)
        new_labels.append(np.asarray(time_dist))


        # uncomment below if doing non Time-Distributed
        # new_labels.append(np.asarray(new_label))

    return np.asarray(new_labels)

def convert_labels_non_time_dist(all_labels, num_classes):
    new_labels = []
    for label in all_labels:
        time_dist = []
        new_label = np.zeros(num_classes)
        new_label[label-1] = 1

        # uncomment below if doing Time-Distributed
        # for i in range(int(num_timesteps/2)):
        #     time_dist.append(new_label)
        # new_labels.append(np.asarray(time_dist))


        # uncomment below if doing non Time-Distributed
        new_labels.append(np.asarray(new_label))

    return np.asarray(new_labels)

def test_model(model, test_x, test_y, batch_size):
    
    predictions = model.predict(test_x, batch_size, verbose=1)

    label_to_tp = {}
    label_to_fp = {}
    label_to_tn = {}
    label_to_fn = {}

    for i in range(num_classes):
        label = i+1
        label_to_tp[label] = 0
        label_to_fp[label] = 0
        label_to_tn[label] = 0
        label_to_fn[label] = 0


    count_correct = 0
    count_wrong = 0
    y_true = []
    y_pred = []
    for pred, actual in zip(predictions,test_y):
        pred_index = pred.argmax(axis=-1)
        act_index = actual.argmax(axis=-1)
        y_pred.append(pred_index+1)
        y_true.append(act_index+1)
        
        #print('Predicted: ' + str(pred_index+1) + ' Actual: ' + str(act_index+1))
        
        if(pred_index == act_index):
            count_correct+=1
            label_to_tp[pred_index+1] +=1
        else:
            label_to_fp[pred_index+1] +=1
            label_to_fn[act_index+1] +=1
            count_wrong+=1
    total = count_correct+count_wrong
    print('Total: ' + str(total))
    print('Correct: ' + str(count_correct))
    print('Wrong: ' + str(count_wrong))

    c_matrix = confusion_matrix(y_true, y_pred)
    all_prec = []
    all_rec = []

    conf = open('conf.txt', 'w+')
    for row in c_matrix:
        conf.write(str(row))
        conf.write('\n')

    # for i in range(len(c_matrix)):
    #     print('For ' + str(get_name_from_label(i+1)) + ': ')
    #     total_guessed_this = sum(c_matrix[i])
    #     true_pos = c_matrix[i][i]
    #     false_pos = total_guessed_this - true_pos
    #     false_neg = 0
    #     for j in range(len(c_matrix)):
    #         if(j==i):
    #             continue
    #         false_neg += c_matrix[i][j]
    #     true_neg = total-false_neg
    #     print('True Pos: ' + str(true_pos))
    #     print('False Pos: ' + str(false_pos))
    #     print('True Neg: ' + str(true_neg))
    #     print('False Neg: ' + str(false_neg))

    #     precision = true_pos / (true_pos + false_pos)
    #     recall = true_pos / (true_pos + false_neg)
    #     all_prec.append(precision)
    #     all_rec.append(recall)

    for i in range(num_classes):
        label = i+1
        print('For ' + str(get_name_from_label(i+1)) + ': ')
        # print('True Pos: ' + str(label_to_tp[label]))
        # print('False Pos: ' + str(label_to_fp[label]))
        # print('False Neg: ' + str(label_to_fn[label]))
        precision = label_to_tp[label] / (label_to_tp[label] + label_to_fp[label])
        recall = label_to_tp[label] / (label_to_tp[label] + label_to_fn[label])
        print(precision)
        print(recall)
        all_prec.append(precision)
        all_rec.append(recall)

    overall_prec = sum(all_prec) / len(all_prec)
    overall_rec = sum(all_rec) / len(all_rec)
    f_one = (2*overall_prec*overall_rec) / (overall_prec + overall_rec)
    print('Overall precision  = ' + str(overall_prec))
    print('Overall recall = ' + str(overall_rec))
    print('Overall F1 score = ' + str(f_one))






def train_model(model, train_x, train_y, experiment_id,
                num_epochs=100, batchsize=64, validation_split=0.1):

    # checkpoint to save the best val_accuracy and val_loss
    acc_path = os.getcwd() + '/best_models/' + model.name + '_experiment' + str(experiment_id) + '_val_acc_{epoch:02d}_{val_categorical_accuracy:.2f}_acc.h5'
    loss_path = os.getcwd() + '/best_models/' + model.name + '_experiment' + str(experiment_id) + '_val_loss_{epoch:02d}_{val_loss:.2f}_loss.h5'
    checkpoint_val_acc = ModelCheckpoint(acc_path, monitor='val_categorical_accuracy', verbose=1, save_best_only=True, mode='max', period=1)
    checkpoint_val_loss = ModelCheckpoint(loss_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min', period=1)
    callbacks_list = [checkpoint_val_loss, checkpoint_val_acc]

    model.fit(train_x, train_y, epochs=15, batch_size=1000, verbose=2,
              callbacks=callbacks_list, validation_split=validation_split, shuffle=True)

def run_single_experiment(experiment, model, num_classes, num_timesteps, experiment_id, num_epochs=100, batchsize=300, validation_split=0.1):

    train_x = experiment.train_x
    #train_y = convert_labels_to_multiclass(experiment.train_y, num_classes, num_timesteps)
    train_y = convert_labels_non_time_dist(experiment.train_y, num_classes)
    #train_x = np.expand_dims(train_x, axis=2)

    test_x = experiment.test_x
    test_y = convert_labels_non_time_dist(experiment.test_y, num_classes)

    print(train_y)
    print(test_y)

    # train the model (saves the best val acc and loss)
    train_model(model, train_x, train_y, experiment_id, num_epochs=num_epochs, batchsize=batchsize,
                validation_split=validation_split)

    test_model(model, test_x, test_y, batch_size=1000)
    

##################################
# *********Main Block************
##################################

########################################
# Training Code to train for each fold
########################################

# load the AllExperiments Object
all_experiments_path = os.getcwd() + '/All_Experiments_Object.pkl'
all_experiments = load_object_pickle(all_experiments_path)

# define important constants
vocab_size = all_experiments.vocab_size
num_timesteps = all_experiments.num_timesteps
embedding_dimension = 4
num_classes = 21
num_conv_filters = 25

print(vocab_size)
print(num_timesteps)
# get the model we want to use
model = create_conv_bidirect_lstm_model(vocab_size, embedding_dimension, num_timesteps, num_conv_filters, num_classes)

experiment_id = 1
# run an experiment
for experiment in all_experiments.all_experiments:
    #get_class_count(experiment.test_y)
    run_single_experiment(experiment, model, num_classes, num_timesteps, experiment_id)
    experiment_id +=1
    break

########################################
# Testing Code to test for each fold
########################################

# model = load_model('C:\\Users\\Royal\\Documents\\GitHub\\Deep_Source_Code_Snippet_Classification\\project\\best_models\\brandon_new_test_experiment1_val_acc_09_0.72_acc.h5')
# experiment = all_experiments.all_experiments[0]
# test_x = experiment.test_x
# test_y = convert_labels_non_time_dist(experiment.test_y, num_classes)

#test_model(model, test_x, test_y, 1000)

# from keras.utils import plot_model

# os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

# plot_model(model, to_file='out_right.png', show_shapes=True, show_layer_names=True, rankdir='LR')