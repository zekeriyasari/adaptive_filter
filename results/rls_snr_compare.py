# Channel equalizer implemented using
# LMS adaptive filtering.


from padapfilt.filters.rls import *
from plotting import *

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
f2 = RLSFilter(m2, w='zeros', lamda=0.98, delta=0.005)

sigma_a = 0.001
sigma_b = 0.1
SNR_a = 10 * np.log10(1.0 / sigma_a)
SNR_b = 10 * np.log10(1.0 / sigma_b)

J_a = np.zeros((l, n))
J_b = np.zeros((l, n))
w_a = np.zeros((l, m2))
w_b = np.zeros((l, m2))
for k in range(l):
    # generate the data.
    x = 2 * np.round(np.random.rand(n + m1 + m2 - 2)) - 1

    # generate the noise.
    v_a = np.sqrt(sigma_a) * np.random.randn(n + m2 - 1)  # 30 dB SNR
    v_b = np.sqrt(sigma_b) * np.random.randn(n + m2 - 1)  # 10 dB SNR

    # filter the data from the channel.
    data_matrix = input_from_history(x, m1)
    u = np.zeros(data_matrix.shape[0])
    for item in range(data_matrix.shape[0]):
        u[item] = f1.estimate(data_matrix[item])
    u_a = u + v_a
    u_b = u + v_b

    # calculate the equalizer output.
    d_vector = x[delay:n + delay:]
    u_matrix_a = input_from_history(u_a, m2)
    u_matrix_b = input_from_history(u_b, m2)

    y_a, e_a, w_a[k] = f2.run(d_vector, u_matrix_a)
    # reset the equalizer for the next trial.
    f2.reset()  # reset the filter to zero tap-weights.
    y_b, e_b, w_b[k] = f2.run(d_vector, u_matrix_b)
    # reset the equalizer for the next trial.
    f2.reset()  # reset the filter to zero tap-weights.

    # calculate learning curve.
    J_a[k] = e_a ** 2
    J_b[k] = e_b ** 2


J_avg_a = J_a.mean(axis=0)
w_avg_a = w_a.mean(axis=0)
J_avg_b = J_b.mean(axis=0)
w_avg_b = w_b.mean(axis=0)

ax1.semilogy(J_avg_a, label=r'$SNR = {}dB$'.format(int(SNR_a)))
ax1.semilogy(J_avg_b, label=r'$SNR = {}dB$'.format(int(SNR_b)))
ax1.legend()

ax2.stem(w_avg_a, label=r'$SNR = {}dB$'.format(int(SNR_a)))
ax2.stem(w_avg_b, label=r'$SNR = {}dB$'.format(int(SNR_b)), markerfmt='D')
ax2.legend()

plt.show()
