import numpy as np
import plotly.graph_objs as go

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def make_metrics(y_true, y_pred):
    def mape():
        return np.mean(np.abs((y_pred - y_true) / y_true)) * 100

    def rmse():
        return np.sqrt(mean_squared_error(y_true, y_pred))

    def mae():
        return mean_absolute_error(y_true, y_pred)

    return {'RMSE': rmse(), 'MAPE': mape(), 'MAE': mae(), 'R2_score': r2_score(y_true, y_pred)}


def plot(y_true=None, y_pred=None, title=''):
    fig = go.Figure()
    fig.add_scatter(y=y_pred, name='Pred')
    fig.add_scatter(y=y_true, name='True')
    fig.update_layout(title=title)
    fig.show()


def split_sequence_multioutput(sequence, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the sequence
        if out_end_ix > len(sequence):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def split_sequence(sequence, n_steps):
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


def createXY(sequence, n_steps=4, n_output=None):
    if n_output is not None:
        X, y = split_sequence_multioutput(sequence=sequence, n_steps_in=n_steps, n_steps_out=n_output)
    else:
        X, y = split_sequence(sequence=sequence, n_steps=n_steps)
    return X, y
