# Keras Models
from keras.models import Sequential

# Keras Layers
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Conv1D
from keras.layers import Bidirectional
from keras.layers import TimeDistributed
from keras.layers import BatchNormalization
from keras.layers import MaxPooling1D
from keras.layers import Flatten
from keras.layers import Dropout

# Keras Regularizers
from keras.regularizers import l2

# Keras Optimizers
from keras.optimizers import Adam
from keras.optimizers import SGD

def create_conv_bidirect_lstm_model(vocab_size, embedding_dimension, num_timesteps, num_conv_filters, num_classes):

    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dimension, input_length=num_timesteps))
    model.add(Conv1D(num_conv_filters, 1, strides=1, activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Bidirectional(LSTM(50, return_sequences=True, kernel_regularizer=l2(0.01))))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(num_classes, activation='softmax')))
    opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0001, amsgrad=False)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['categorical_accuracy'], sample_weight_mode="temporal")

    model.name = 'embed_conv_mp_bilstm_dense_adam'

    model.summary()

    return model

def create_simple_lstm_model(vocab_size, embedding_dimension, num_timesteps, num_classes):

    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dimension, input_length=num_timesteps))
    model.add(LSTM(100))
    model.add(Dense(num_classes, activation='softmax'))
    opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0001, amsgrad=False)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['categorical_accuracy'], sample_weight_mode="temporal")

    model.name = 'embed_lstm_dense_adam'

    model.summary()

    return model

def save_model(model):

    #get the model's name
    name = model.name

    #save the whole model (architecture, weights, optimizer)
    model.save(name + '.h5')

def load_model(model_path):

    model = load_model(model_path)

    print('Loaded: ' + model.name)
    print('Model''s Architecture: ')
    model.summary()

    return model





