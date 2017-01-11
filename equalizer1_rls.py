# Channel equalizer implemented using
# RLS adaptive filtering.


from padapfilt.filters.rls import *
import matplotlib.pyplot as plt

# determine simulation parameters.
n = 5000  # number of input data samples to the equalizer.
m1 = 3  # number of taps of channel.
m2 = 21  # number of taps of equalizer
l = 100  # number of trials.
delay = int(m1 / 2) + int(m2 / 2)

# try the system four channel models.
channels = np.array([[0.25, 1.0, 0.25],
                     [0.25, 1.0, -0.25],
                     [-0.25, 1.0, 0.25]])

# take two figures for the plots
fig1, ax1 = plt.subplots(1)  # plots the learning curves.
fig2, ax2 = plt.subplots(channels.shape[0])  # plots found filter taps.

for i in range(channels.shape[0]):
    # construct the channel.
    h = channels[i]

    # construct the channel filter
    f1 = BaseFilter(m1, w=h)

    # construct the equalizer.
    f2 = RLSFilter(m2, w='zeros', delta=0.005, lamda=0.98)

    J = np.zeros((l, n))
    w = np.zeros((l, m2))
    for k in range(l):
        # generate the data.
        x = 2 * np.round(np.random.rand(n + m1 + m2 - 2)) - 1

        # generate the noise.
        v = np.sqrt(0.01) * np.random.randn(n + m2 - 1)

        # filter the data from the channel.
        data_matrix = input_from_history(x, m1)
        u = np.zeros(data_matrix.shape[0])
        for item in range(data_matrix.shape[0]):
            u[item] = f1.estimate(data_matrix[item])
        u += v

        # calculate the equalizer output.
        d_vector = x[delay:n + delay:]

        u_matrix = input_from_history(u, m2)
        y, e, w[k] = f2.run(d_vector, u_matrix)

        # calculate learning curve.
        J[k] = e ** 2

    J_avg = J.mean(axis=0)
    w_avg = w.mean(axis=0)
    ax1.semilogy(J_avg, label='$H_{}$'.format(i))
    ax2[i].stem(w_avg, label='$H_{}$'.format(i))


ax1.legend()
plt.show()
