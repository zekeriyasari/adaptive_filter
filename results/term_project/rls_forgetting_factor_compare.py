# Performance comparison of RLS channel equalizer
# for different forgetting factors.


from padapfilt.filters.rls import *

# determine simulation parameters.
n = 2000  # number of input data samples to the equalizer.
m1 = 3  # number of taps of channel.
m2 = 21  # number of taps of equalizer
l = 100  # number of trials.
delay = int(m1 / 2) + int(m2 / 2)

# try the channel model.
channel = np.array([0.25, 1.0, -0.25])

# take two figures for the plots
fig1, ax1 = get_learning_curve_plot()  # plots the learning curves.
fig2, ax2 = get_tap_weights_graph(1)  # plots found filter taps.

# construct the channel.
h = channel

# construct the channel filter
f1 = BaseFilter(m1, w=h)

# construct two equalizer.
lambda_0 = 0.98
lambda_1 = 0.85
f2_a = RLSFilter(m2, w='zeros', lamda=lambda_0, delta=0.005)
f2_b = RLSFilter(m2, w='zeros', lamda=lambda_1, delta=0.005)

J_a = np.zeros((l, n))
J_b = np.zeros((l, n))
w_a = np.zeros((l, m2))
w_b = np.zeros((l, m2))
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

    # calculate the equalizer output.
    d_vector = x[delay:n + delay:]
    u_matrix = input_from_history(u, m2)

    y_a, e_a, w_a[k] = f2_a.run(d_vector, u_matrix)
    y_b, e_b, w_b[k] = f2_b.run(d_vector, u_matrix)

    # calculate learning curve.
    J_a[k] = e_a ** 2
    J_b[k] = e_b ** 2

    # reset the equalizer for the next trial.
    f2_a.reset()  # reset the filter to zero tap-weights.
    f2_b.reset()  # reset the filter to zero tap-weights.

J_avg_a = J_a.mean(axis=0)
w_avg_a = w_a.mean(axis=0)
J_avg_b = J_b.mean(axis=0)
w_avg_b = w_b.mean(axis=0)

ax1.semilogy(J_avg_a, label=r'$\lambda = {}$'.format(lambda_0))
ax1.semilogy(J_avg_b, label=r'$\lambda = {}$'.format(lambda_1))
ax1.legend()

ax2.stem(w_avg_a, label=r'$\lambda = {}$'.format(lambda_0))
ax2.stem(w_avg_b, label=r'$\lambda = {}$'.format(lambda_1), markerfmt='D')
ax2.legend()

plt.show()
