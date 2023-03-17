from keras.layers import Input, Dense, LSTM, Conv1D, Dropout, Bidirectional, Multiply
from keras.models import Model
import keras.backend as K
import numpy as np
from ..model import BaseModel
from ...metric import r2_keras

#from attention_utils import get_activations
# from keras.layers import merge
from keras.layers import Multiply
from keras.layers.core import *
# from keras.layers.recurrent import LSTM
from keras.models import *

import pandas as pd
import numpy as np

SINGLE_ATTENTION_VECTOR = False


def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = inputs
    #a = Permute((2, 1))(inputs)
    #a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(input_dim, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((1, 2), name='attention_vec')(a)

    #output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul


# Another way of writing the attention mechanism is suitable for the use of the above error source:https://blog.csdn.net/uhauha2929/article/details/80733255
def attention_3d_block2(inputs, single_attention_vector=False):
    # If the upper layer is LSTM, you need return_sequences=True
    # inputs.shape = (batch_size, time_steps, input_dim)
    time_steps = K.int_shape(inputs)[1]
    input_dim = K.int_shape(inputs)[2]
    a = Permute((2, 1))(inputs)
    a = Dense(time_steps, activation='softmax')(a)
    if single_attention_vector:
        a = Lambda(lambda x: K.mean(x, axis=1))(a)
        a = RepeatVector(input_dim)(a)

    a_probs = Permute((2, 1))(a)
    # Multiplied by the attention weight, but there is no summation, it seems to have little effect
    # If you classify tasks, you can do Flatten expansion
    # element-wise
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul


def create_dataset(dataset, look_back):
    """
    Processing the data
    """
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back, :])
    TrainX = np.array(dataX)
    Train_Y = np.array(dataY)

    return TrainX, Train_Y


# Multidimensional normalization returns data and maximum and minimum values
def NormalizeMult(data):
    #normalize Used for denormalization
    data = np.array(data)
    normalize = np.arange(2 * data.shape[1], dtype='float64')

    normalize = normalize.reshape(data.shape[1], 2)
    print(normalize.shape)
    for i in range(0, data.shape[1]):
        #Column i
        list = data[:, i]
        listlow, listhigh = np.percentile(list, [0, 100])
        # print(i)
        normalize[i, 0] = listlow
        normalize[i, 1] = listhigh
        delta = listhigh - listlow
        if delta != 0:
            #Row j
            for j in range(0, data.shape[0]):
                data[j, i] = (data[j, i] - listlow) / delta
    #np.save("./normalize.npy",normalize)
    return data, normalize


# Multidimensional denormalization
def FNormalizeMult(data, normalize):
    data = np.array(data)
    for i in range(0, data.shape[1]):
        listlow = normalize[i, 0]
        listhigh = normalize[i, 1]
        delta = listhigh - listlow
        if delta != 0:
            #Row j
            for j in range(0, data.shape[0]):
                data[j, i] = data[j, i] * delta + listlow

    return data


class CNN_BiLSTM_Attention(BaseModel):

    def __init__(self, n_steps=1, n_output=1, n_seq=None, **kwargs):
        print('构建CNN_BiLSTM_Attention...')
        super().__init__(n_steps=n_steps, n_output=n_output, n_seq=None, **kwargs)

    def build_model(self, layer_size=16, learning_rate=3e-3, n_steps=4,
                    filters=64, kernel_size=1, pool_size=2, dropout=0.3, **kwargs):
        inputs = Input(shape=(self.n_steps, self.n_features))

        x = Conv1D(filters=filters, kernel_size=1, activation='relu')(inputs)  # , padding = 'same'
        x = Dropout(0.3)(x)

        # lstm_out = Bidirectional(LSTM(lstm_units, activation='relu'), name='bilstm')(x)
        # For GPU you can use CuDNNLSTM
        lstm_out = Bidirectional(LSTM(layer_size, return_sequences=True))(x)
        lstm_out = Dropout(dropout)(lstm_out)
        attention_mul = attention_3d_block(lstm_out)
        attention_mul = Flatten()(attention_mul)

        # output = Dense(1, activation='sigmoid')(attention_mul)
        output = Dense(1, activation='linear')(attention_mul)
        model = Model(inputs=[inputs], outputs=output)
        model.compile(loss='mse', optimizer='adam', metrics=['mae', r2_keras])
        return model

