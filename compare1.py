# Channel equalizer RLS and LMS performance comparing.

from padapfilt.filters.lms import *
from padapfilt.filters.rls import *
from plotting import *


def raised_cos(x_in, w_in=2.9):
    """
    Raised cosine inter-symbol interference channel model.
    """
    return 0.5 * (1 + np.cos(2 * np.pi / w_in * (x_in - 2)))


# determine simulation parameters.
n = 5000  # number of input data samples to the equalizer.
m1 = 3  # number of taps of channel.
m2 = 11  # number of taps of equalizer
l = 100  # number of trials.
delay = int(m1 / 2) + int(m2 / 2)

# try the system for different  channel models.
channels = np.array([[-0.25, 1.0, 0.25]])

# take two figures for the plots
fig1, ax1 = get_learning_curve_plot()  # plots the learning curves.
fig2, ax2 = get_tap_weights_graph(2)  # plots found filter taps.

for i in range(channels.shape[0]):
    # construct the channel.
    h = channels[i]

    # construct the channel filter
    f1 = BaseFilter(m1, w=h)

    # construct the equalizer with lms filter.
    f_lms = LMSFilter(m2, mu=0.0, w='zeros')

    # construct the equalizer with lms filter.
    f_rls = RLSFilter(m2, w='zeros', delta=0.005, lamda=0.98)

    J_lms = np.zeros((l, n))
    w_lms = np.zeros((l, m2))
    J_rls = np.zeros((l, n))
    w_rls = np.zeros((l, m2))
    for k in range(l):
        # generate the data.
        x = 2 * np.round(np.random.rand(n + m1 + m2 - 2)) - 1

        # generate the noise.
        v = np.sqrt(0.1) * np.random.randn(n + m2 - 1)

        # filter the data from the channel.
        data_matrix = input_from_history(x, m1)
        u = np.zeros(data_matrix.shape[0])
        for item in range(data_matrix.shape[0]):
            u[item] = f1.estimate(data_matrix[item])
        u += v

        u_matrix = input_from_history(u, m2)

        # calculate the equalizer output.
        d_vector = x[delay:n + delay:]

        y_lms, e_lms, w_lms[k] = f_lms.run(d_vector, u_matrix)
        y_rls, e_rls, w_rls[k] = f_rls.run(d_vector, u_matrix)

        # calculate learning curve.
        J_lms[k] = e_lms ** 2
        J_rls[k] = e_rls ** 2

        # reset the equalizers for the next trial.
        f_lms.reset()  # reset the filter to zero tap-weights.
        f_rls.reset()  # reset the filter to zero tap-weights.

    J_lms_avg = J_lms.mean(axis=0)
    w_lms_avg = w_lms.mean(axis=0)
    J_rls_avg = J_rls.mean(axis=0)
    w_rls_avg = w_rls.mean(axis=0)

    ax1.semilogy(J_lms_avg, label='$LMS$')
    ax1.semilogy(J_rls_avg, label='$RLS$')
    ax1.legend()

    ax2[0].stem(w_rls_avg, label='$LMS$')
    ax2[0].legend()

    ax2[1].stem(w_rls_avg, label='$RLS$')
    ax2[1].legend()

plt.show()
