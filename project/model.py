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

# Keras Optimizers
from keras.optimizers import Adam
from keras.optimizers import SGD

def get_conv_bidirect_lstm_model(vocab_size, embedding_dimension, num_timesteps, num_conv_filters):

    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dimension, input_length=num_timesteps))
    model.add(Conv1D(num_conv_filters, 1, strides=1, activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Bidirectional(LSTM(300, return_sequences=True)))
    model.add(TimeDistributed(Dense(5, activation='softmax')))
    opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0001, amsgrad=False)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['categorical_accuracy'], sample_weight_mode="temporal")

    model.summary()

    return model

