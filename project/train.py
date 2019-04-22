import os
import numpy as np

from dataset import Dataset
from dataset import DataPoint
from dataset import AllExperiments
from dataset import SingleExperiment

from model import create_simple_lstm_model
from model import create_conv_bidirect_lstm_model
from model import save_model
from model import load_model

from tokenize_data import save_object_pickle
from tokenize_data import load_object_pickle

from keras.callbacks import ModelCheckpoint

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

# def test_model(model, test_x, test_y):





def train_model(model, train_x, train_y, experiment_id,
                num_epochs=100, batchsize=64, validation_split=0.1):

    # checkpoint to save the best val_accuracy and val_loss
    acc_path = os.getcwd() + '/best_models/' + model.name + '_experiment' + str(experiment_id) + '_val_acc_{epoch:02d}_{val_categorical_accuracy:.2f}_acc.h5'
    loss_path = os.getcwd() + '/best_models/' + model.name + '_experiment' + str(experiment_id) + '_val_loss_{epoch:02d}_{val_loss:.2f}_loss.h5'
    checkpoint_val_acc = ModelCheckpoint(acc_path, monitor='val_categorical_accuracy', verbose=1, save_best_only=True, mode='max', period=1)
    checkpoint_val_loss = ModelCheckpoint(loss_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min', period=1)
    callbacks_list = [checkpoint_val_loss, checkpoint_val_acc]

    model.fit(train_x, train_y, epochs=num_epochs, batch_size=1000, verbose=2,
              callbacks=callbacks_list, validation_split=validation_split, shuffle=True)

def run_single_experiment(experiment, model, num_classes, num_timesteps, experiment_id, num_epochs=100, batchsize=300, validation_split=0.1):

    train_x = experiment.train_x
    train_y = convert_labels_to_multiclass(experiment.train_y, num_classes, num_timesteps)
    #train_x = np.expand_dims(train_x, axis=2)

    test_x = experiment.test_x
    test_y = experiment.test_y

    # train the model (saves the best val acc and loss)
    train_model(model, train_x, train_y, experiment_id, num_epochs=num_epochs, batchsize=batchsize,
                validation_split=validation_split)
    
    # test the model


##################################
# *********Main Block************
##################################

# load the AllExperiments Object
all_experiments_path = os.getcwd() + '/All_Experiments_Object.pkl'
all_experiments = load_object_pickle(all_experiments_path)

# define important constants
vocab_size = all_experiments.vocab_size
num_timesteps = all_experiments.num_timesteps
embedding_dimension = 4
num_classes = 21
num_conv_filters = 25

# get the model we want to use
model = create_conv_bidirect_lstm_model(vocab_size, embedding_dimension, num_timesteps, num_conv_filters, num_classes)

experiment_id = 1
# run an experiment
for experiment in all_experiments.all_experiments:
    #get_class_count(experiment.test_y)
    run_single_experiment(experiment, model, num_classes, num_timesteps, experiment_id)
    experiment_id +=1