import numpy as np
import pandas as pd
import plotly.graph_objs as go

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.utils import check_array


def make_metrics(y_true, y_pred):
    """
    To calculate the score or error in time series forcast tasks
    :param y_true: the real value
    :param y_pred: to predict value
    :return: a dict, with rmse, mape, mae, r2
    """

    def mape():
        return np.mean(np.abs((y_pred - y_true) / y_true)) * 100

    def rmse():
        return np.sqrt(mean_squared_error(y_true, y_pred))

    def mae():
        return mean_absolute_error(y_true, y_pred)

    return {'RMSE': rmse(), 'MAPE': mape(), 'MAE': mae(), 'R2_score': r2_score(y_true, y_pred)}


def plot(y_true=None, y_pred=None, title=''):
    """
    Plot the real value and predict value fitting curve use plot(https://plotly.com/python/)
    :param y_true:
    :param y_pred:
    :param title:
    :return:
    """
    fig = go.Figure()
    fig.add_scatter(y=y_pred, name='Pred')
    fig.add_scatter(y=y_true, name='True')
    fig.update_layout(title=title)
    fig.show()


def split_sequence_multioutput(sequence, n_steps_in, n_steps_out, label_idx=-1):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the sequence
        if out_end_ix > len(sequence):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix, :], sequence[end_ix:out_end_ix, label_idx]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def split_sequence(sequence, n_steps, label_idx=-1):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def check_input(data=None):
    """
    Check input and transform into numpy.array
    :param data:
    :return: numpy.array
    """
    if data is None:
        raise ValueError("Input must not be None")

    data = check_array(data)
    return np.array(data)


def createXY(sequence, n_steps=4, n_output=1, label_idx=-1):
    """
    Split origin data into X and y. For multivariate task, should specify which column is the label column,
    default is the last column (-1)
    :param sequence:
    :param n_steps:
    :param n_output:
    :param label_idx:
    :return:
    """
    # transform input data into np.array
    sequence = check_input(sequence)
    # calculate the sequence dims
    n_dim = np.ndim(sequence)
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        out_end_ix = end_ix + n_output
        # check if we are beyond the sequence
        if out_end_ix > len(sequence) - 1:
            break
        # gather input and output parts of the pattern
        if n_dim == 1:
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        else:
            seq_x, seq_y = sequence[i:end_ix, :sequence.shape[1]], sequence[end_ix:out_end_ix, label_idx]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

