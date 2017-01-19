import numpy as np
import matplotlib.pyplot as plt


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


def get_learning_curve_plot():
    fig, ax = plt.subplots()
    ax.grid(which='both')
    ax.set_xlabel(r'$Number \; of \; iterations, \; n$')
    ax.set_ylabel(r'$Ensemble-averaged \; square \; error$')
    plt.tight_layout()
    return fig, ax


def get_tap_weights_graph(n):
    if n:
        if n > 1:
            fig, ax = plt.subplots(n)
            for i in range(n):
                ax[i].set_xlabel(r'$k$')
                ax[i].set_ylabel(r'$\hat{w}_k$')
                ax[i].set_ylim([-2, 2])
        else:
            fig, ax = plt.subplots(1)
            ax.set_xlabel(r'$k$')
            ax.set_ylabel(r'$\hat{w}_k$')
            ax.set_ylim([-2, 2])
    else:
        fig, ax = plt.subplots(1)
        ax.set_xlabel(r'$k$')
        ax.set_ylabel(r'$\hat{w}_k$')
        ax.set_ylim([-2, 2])
    return fig, ax

