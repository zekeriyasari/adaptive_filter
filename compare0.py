# Channel equalizer RLS and LMS performance comparing.


from padapfilt.filters.lms import *
from padapfilt.filters.rls import *
import matplotlib.pyplot as plt


def raised_cos(x_in, w_in=2.9):
    """
    Raised cosine inter-symbol interference channel model.
    """
    return 0.5 * (1 + np.cos(2 * np.pi / w_in * (x_in - 2)))

# determine simulation parameters.
n = 500  # number of input data samples to the equalizer.
m1 = 5  # number of taps of channel.
m2 = 11  # number of taps of equalizer
l = 200  # number of trials.
delay = int(m1 / 2) + int(m2 / 2)

# try the system four channel models.
# omega = np.array([2.9, 3.1, 3.3, 3.5])
omega = np.array([3.1])

# take two figures for the plots
fig1, ax1 = plt.subplots(1)  # plots the learning curves.
fig2, ax2 = plt.subplots(len(omega))  # plots found filter taps.
fig3, ax3 = plt.subplots(len(omega))  # plots found filter taps.

for i in range(len(omega)):
    # construct the channel.
    h = np.array([y for y in map(lambda t: raised_cos(t, w_in=omega[i]) if 1 <= t <= 3 else 0,
                                 np.arange(m1))])  # channel impulse response

    # construct the channel filter
    f1 = BaseFilter(m1, w=h)

    # construct the equalizer with lms filter.
    f_lms = LMSFilter(m2, mu=0.075, w='zeros')

    # construct the equalizer with lms filter.
    f_rls = RLSFilter(m2, w='zeros', delta=0.004, lamda=0.98)

    J_lms = np.zeros((l, n))
    w_lms = np.zeros((l, m2))
    J_rls = np.zeros((l, n))
    w_rls = np.zeros((l, m2))
    for k in range(l):
        # generate the data.
        x = 2 * np.round(np.random.rand(n + m1 + m2 - 2)) - 1

        # generate the noise.
        v = np.sqrt(0.001) * np.random.randn(n + m2 - 1)

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

    J_lms_avg = J_lms.mean(axis=0)
    w_lms_avg = w_lms.mean(axis=0)
    J_rls_avg = J_rls.mean(axis=0)
    w_rls_avg = w_rls.mean(axis=0)

    ax1.semilogy(J_lms_avg, label='$H_{}$'.format(i))
    ax1.semilogy(J_rls_avg, label='$H_{}$'.format(i))
    ax2.stem(w_rls_avg, label='$H_{}$'.format(i))
    ax3.stem(w_rls_avg, label='$H_{}$'.format(i))

ax1.legend()
plt.show()
