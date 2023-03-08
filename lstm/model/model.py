import os
import time

import keras
import pandas as pd
from keras.layers import LSTM, Flatten, TimeDistributed, Bidirectional, Dropout
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.optimizers import Adam
from attention import Attention

import lstm.util as util


class BaseModel:
    def __init__(self, n_steps=1, n_output=1, n_features=1, n_seq=None, **kwargs):
        self.y_hat = None
        self.n_features = None
        self.X_train = None
        self.X_test = None
        self.n_seq = n_seq
        self.n_steps = n_steps
        self.n_output = n_output
        self.n_features = n_features
        self.has_predicted = False
        self.model_name = self.__class__.__name__
        self.model = self.build_model(**kwargs)
        print('构建完毕！')

    def createXY(self, sequence, test_size=0.2, return_X_y=False):

        X, y = util.createXY(sequence, n_steps=self.n_steps, n_output=self.n_output)
        if self.n_seq is not None:
            X = X.reshape((X.shape[0], self.n_seq, int(self.n_steps / self.n_seq), self.n_features))
        if 0 < test_size < 1:
            split_index = int(len(X) * (1 - test_size))
        else:
            split_index = len(X) - test_size
        self.X_train, self.y_train, self.X_test, self.y_test = X[:split_index], y[:split_index], \
            X[split_index:], y[split_index:]
        if return_X_y:
            return self.X_train, self.y_train, self.X_test, self.y_test

    def fit(self, X=None, y=None, **kwargs):
        if X is None or y is None:
            self.model.fit(self.X_train, self.y_train, **kwargs)
        else:
            self.model.fit(X, y, **kwargs)

    def predict(self, X=None):
        if X is None:
            self.y_hat = self.model.predict(self.X_test)
            self.has_predicted = True
            return self.y_hat
        else:
            return self.model.predict(self.X_test)

    def score(self, y_true=None, y_pred=None):
        if y_true is None or y_pred is None:
            if not self.has_predicted:
                self.predict()
            return util.make_metrics(self.y_test, self.y_hat)
        else:
            return util.make_metrics(y_true, y_pred)

    def build_model(self):
        pass

    def summary(self):
        return self.model.summary()

    def get_model(self):
        return self.model

    def get_name(self):
        return self.model_name

    def save_metric(self, path='', file_type='csv'):

        current_time = (lambda: int(round(time.time() * 1000)))()
        alpha_start = 65
        if self.n_output > 1:
            df = pd.DataFrame(self.y_test, columns=[chr(i) for i in range(alpha_start, alpha_start + self.n_output)])
            file_name = os.path.join(path, '{}-{}-{}-True'.format(self.model_name, self.n_output, current_time))
            if file_type == 'csv':
                df.to_csv(file_name + '.csv')
            elif file_type == 'xlsx':
                df.to_excel(file_name + '.xlsx')
            else:
                raise Exception('文件类型不支持，请设置 file_type=csv | xlsx')

            df = pd.DataFrame(self.y_hat, columns=[chr(i) for i in range(alpha_start, alpha_start + self.n_output)])
            file_name = os.path.join(path, '{}-{}-{}-True'.format(self.model_name, self.n_output, current_time))
            if file_type == 'csv':
                df.to_csv(file_name + '.csv')
            elif file_type == 'xlsx':
                df.to_excel(file_name + '.xlsx')

        else:
            df = pd.concat([pd.Series(self.y_test), pd.Series(self.y_hat)], names=['True', 'Pred'], axis=1)
            file_name = os.path.join(path, '{}-{}'.format(self.model_name, current_time))
            if file_type == 'csv':
                df.to_csv(file_name + '.csv')
            elif file_type == 'xlsx':
                df.to_excel(file_name + '.xlsx')
            else:
                raise Exception('文件类型不支持，请设置 file_type=csv | xlsx')


class LSTM(BaseModel):

    def __init__(self, n_steps=1, n_output=1, **kwargs):
        print('构建LSTM...')
        super().__init__(n_steps=n_steps, n_output=n_output, **kwargs)

    def build_model(self, layer_size=16, learning_rate=3e-3, **kwargs):
        lstm = keras.models.Sequential()
        lstm.add(keras.layers.LSTM(layer_size, activation='relu', input_shape=(self.n_steps, self.n_features)))
        lstm.add(keras.layers.Dense(self.n_output))
        optimizer = keras.optimizers.Adam(learning_rate)
        lstm.compile(optimizer=optimizer, loss='mse')
        return lstm


class BiLSTM(BaseModel):

    def __init__(self, n_steps=1, n_output=1, **kwargs):
        print('构建BiLSTM...')
        super().__init__(n_steps=n_steps, n_output=n_output, **kwargs)

    def build_model(self, layer_size=16, learning_rate=3e-3, **kwargs):
        bilstm = keras.models.Sequential()
        bilstm.add(Bidirectional(keras.layers.LSTM(layer_size, activation='relu',
                                                   input_shape=(self.n_steps, self.n_features))))
        bilstm.add(keras.layers.Dense(self.n_output))
        optimizer = keras.optimizers.Adam(learning_rate)
        bilstm.compile(optimizer=optimizer, loss='mse')
        return bilstm

    def createXY(self, sequence, n_steps=4, test_size=0.2, return_XY=False):

        X, y = util.createXY(sequence, n_steps=n_steps, n_output=self.n_output)
        X = X.reshape((X.shape[0], X.shape[1], self.n_features))
        if 0 < test_size < 1:
            split_index = int(len(X) * (1 - test_size))
        else:
            split_index = len(X) - test_size
        self.X_train, self.y_train, self.X_test, self.y_test = X[:split_index], y[:split_index], \
            X[split_index:], y[split_index:]
        if return_XY:
            return self.X_train, self.y_train, self.X_test, self.y_test


class CNN_LSTM(BaseModel):

    def __init__(self, n_steps=1, n_output=1, n_seq=None, **kwargs):
        print('构建CNN_LSTM...')
        super().__init__(n_steps=n_steps, n_output=n_output, **kwargs)

    def build_model(self, layer_size=16, learning_rate=3e-3, n_steps=4,
                    filters=64, kernel_size=1, pool_size=2, **kwargs):
        cnn_lstm = keras.models.Sequential()
        cnn_lstm.add(TimeDistributed(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu',
                                            input_shape=(None, n_steps, self.n_features))))
        cnn_lstm.add(TimeDistributed(MaxPooling1D(pool_size=pool_size)))
        cnn_lstm.add(TimeDistributed(Flatten()))
        cnn_lstm.add(keras.layers.LSTM(50, activation='relu'))
        cnn_lstm.add(keras.layers.Dense(self.n_output))
        optimizer = keras.optimizers.Adam(learning_rate)
        cnn_lstm.compile(optimizer=optimizer, loss='mse')
        return cnn_lstm


class CNN_BiLSTM(BaseModel):

    def __init__(self, n_steps=1, n_output=1, n_seq=None, **kwargs):
        print('构建CNN_BiLSTM...')
        super().__init__(n_steps=n_steps, n_output=n_output, n_seq=None, **kwargs)

    def build_model(self, layer_size=16, learning_rate=3e-3, n_steps=4,
                    filters=64, kernel_size=1, pool_size=2, dropout=0.1, **kwargs):
        cnn_bilstm = keras.models.Sequential()
        cnn_bilstm.add(TimeDistributed(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu',
                                              input_shape=(None, n_steps, self.n_features))))
        cnn_bilstm.add(TimeDistributed(MaxPooling1D(pool_size=pool_size)))
        cnn_bilstm.add(TimeDistributed(Flatten()))
        cnn_bilstm.add(Bidirectional(keras.layers.LSTM(layer_size, activation='relu',
                                                       input_shape=(self.n_steps, self.n_features))))
        cnn_bilstm.add(Dropout(dropout))
        cnn_bilstm.add(keras.layers.Dense(self.n_output))
        optimizer = keras.optimizers.Adam(learning_rate)
        cnn_bilstm.compile(optimizer=optimizer, loss='mse')
        return cnn_bilstm


class CNN_BiLSTM_Attention(BaseModel):

    def __init__(self, n_steps=1, n_output=1, n_seq=None, **kwargs):
        print('构建AC_LSTM...')
        super().__init__(n_steps=n_steps, n_output=n_output, n_seq=n_seq, **kwargs)

    def build_model(self, hidden_layers=1, layer_size=16, learning_rate=3e-3, n_steps=4,
                    filters=64, kernel_size=1, pool_size=2, dropout=0.1, **kwargs):
        ac_lstm = keras.models.Sequential()
        ac_lstm.add(TimeDistributed(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'),
                                    input_shape=(None, int(self.n_steps / self.n_seq), self.n_features)))
        ac_lstm.add(TimeDistributed(MaxPooling1D(pool_size=pool_size)))
        ac_lstm.add(TimeDistributed(Flatten()))
        ac_lstm.add(Bidirectional(keras.layers.LSTM(layer_size, activation='relu', return_sequences=True)))
        ac_lstm.add(Dropout(dropout))
        for _ in range(hidden_layers - 1):
            ac_lstm.add(Bidirectional(keras.layers.LSTM(layer_size, activation='relu', return_sequences=True)))
            ac_lstm.add(Dropout(dropout))
        ac_lstm.add(Attention(units=layer_size * 2))
        ac_lstm.add(keras.layers.Dense(self.n_output))
        optimizer = keras.optimizers.Adam(learning_rate)
        ac_lstm.compile(optimizer=optimizer, loss='mse')
        return ac_lstm
