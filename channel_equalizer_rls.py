# Channel equalizer implemented using
# LMS adaptive filtering.


from padapfilt.filters.rls import *
import matplotlib.pyplot as plt


def raised_cos(x_in, w_in=2.9):
    """
    Raised cosine inter-symbol interference channel model.
    """
    return 0.5 * (1 + np.cos(2 * np.pi / w_in * (x_in - 2)))

# determine simulation parameters.
n = 200  # number of input data samples to the equalizer.
m1 = 4  # number of taps of channel.
m2 = 11  # number of taps of equalizer
l = 200  # number of trials.
delay = int(m1 / 2) + int(m2 / 2)

# try the system four channel models.
# omega = np.array([2.9, 3.1, 3.3, 3.5])
omega = np.array([3.1])

# take two figures for the plots
fig1, ax1 = plt.subplots(1)  # plots the learning curves.
# fig2, ax2 = plt.subplots(len(omega))  # plots found filter taps.

for i in range(len(omega)):
    # construct the channel.
    h = np.array([y for y in map(lambda t: raised_cos(t, w_in=omega[i]) if 1 <= t <= 3 else 0,
                                 np.arange(m1))])  # channel impulse response

    # construct the channel filter
    f1 = BaseFilter(m1, w=h)

    # construct the equalizer.
    f2 = RLSFilter(m2, w='zeros', delta=0.004, lamda=0.98)

    J = np.zeros((l, n))
    w = np.zeros((l, m2))
    for k in range(l):
        # generate the data.
        x = 2 * np.random.binomial(1, 0.5, n + m1 + m2 - 2) - 1

        # generate the noise.
        v = np.random.normal(scale=0.001, size=n + m2 - 1)

        # filter the data from the channel.
        data_matrix = input_from_history(x, m1)
        u = np.zeros(data_matrix.shape[0])
        for item in range(data_matrix.shape[0]):
            u[item] = f1.estimate(data_matrix[item])
        u += v

        # calculate the equalizer output.
        d_vector = x[delay:n + m1 + m2 - 1 - delay:]

        u_matrix = input_from_history(u, m2)
        y, ksi, w[k] = f2.run(d_vector, u_matrix)

        # calculate learning curve.
        J[k] = ksi ** 2

    J_avg = J.mean(axis=0)
    # w_avg = w.mean(axis=0)
    ax1.plot(J_avg, label='$H_{}$'.format(i))
    # ax2[i].stem(w_avg, label='$H_{}$'.format(i))


plt.grid(which='both')
ax1.legend()
plt.show()
