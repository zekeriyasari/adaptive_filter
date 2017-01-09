import numpy as np


def input_from_history(data_sequence, m, bias=False):
    assert type(m) == int and m > 0, 'The argument m must be positive int.'
    assert type(data_sequence) == np.ndarray, 'The data_sequence must be numpy array.'

    u_matrix = np.array([data_sequence[i:i + m] for i in range(data_sequence.size - m + 1)])
    if bias:
        u_matrix = np.vstack((u_matrix.T, np.ones(len(u_matrix)))).T
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


def raised_cos(x_in, w_in=2.9):
    """
    Raised cosine inter-symbol interference channel model.
    """
    return 0.5 * (1 + np.cos(2 * np.pi / w_in * (x_in - 2)))


def fir_filter(x, w):
    """
    An FIR filter.
    :param x: input matrix (N-by-M) where M is the number of filter taps.
    :param w: filter coefficients.
    :return: filtered output.
    """

    n = len(x)
    y = np.zeros(n)
    for k in range(n):
        y[k] = np.dot(w, x[k])
    return y


def init_weights(w, m):
    assert type(w) == str or len(w) == m, 'Could not understand w'
    if w == 'random':
        w = np.random.normal(0, 0.5, m)
    elif w == 'zeros':
        w = np.zeros(m)
    else:
        try:
            w = np.array(w)
        except:
            raise ValueError('w must be a list or numpy.array')
    return w


def check_float_range(param, lower_limit, upper_limit, name):
    try:
        param = float(param)
    except:
        raise ValueError('Parameter {}  is not float'.format(name))
    if lower_limit is not None or upper_limit is not None:
        if not lower_limit <= param <= upper_limit:
            raise ValueError('Parameter {} is not in range <{}, {}>'.format(name, lower_limit, upper_limit))
    return param


def check_int_range(param, lower_limit, upper_limit, name):
    try:
        param = int(param)
    except:
        raise ValueError('Parameter {}  is not int'.format(name))
    if lower_limit is not None or upper_limit is not None:
        if not lower_limit <= param <= upper_limit:
            raise ValueError('Parameter {} is not in range <{}, {}>'.format(name, lower_limit, upper_limit))
    return param


def check_int(param, error_msg):
    assert type(param) == int, error_msg
    return param
