import numpy as np


def input_from_history(data_sequence, m):
    """
    Creates the input n-by-m data_matrix from the data_sequence vector
    :param data_sequence: ndarray,
        data_sequence vector
    :param m: int,
        number of columns in the data_matrix, i.ksi
    :return:
    """
    assert type(m) == int and m > 0, 'The argument m must be positive int.'
    assert type(data_sequence) == np.ndarray, 'The data_sequence must be numpy array.'

    u_matrix = np.array([data_sequence[i:i + m][::-1] for i in range(data_sequence.size - m + 1)])
    return u_matrix


def acf(x, l):
    """
    Auto-correlation function.

    :param x: input vector
    :param l: lag
    :return: auto-correlation value.
    """
    y = np.roll(x, l)
    y[:l] = 0
    return np.correlate(x, y, mode='valid')

