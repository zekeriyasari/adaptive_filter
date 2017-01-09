# This module contains some of the
# utility functions used in the project.


import numpy as np


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
