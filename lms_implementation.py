# Channel equalizer implemented using
# LMS adaptive filtering.


from utils import *
from padasip import input_from_history, FilterLMS
import matplotlib.pyplot as plt


# determine simulation parameters.
N = 500  # number of input data samples to the equalizer.
M1 = 4  # number of taps of channel.
M2 = 11  # number of taps of equalizer
L = 200  # number of trials.
delay = int(M1 / 2) + int(M2 / 2)

# try the system four channel models.
omega = np.array([2.9, 3.1, 3.3, 3.5])

# take two figures for the plots
fig1, ax1 = plt.subplots(1)  # plots the learning curves.
fig2, ax2 = plt.subplots(len(omega))  # plots found filter taps.

for i in range(len(omega)):
    # construct the channel.
    h = np.array([y for y in map(lambda t: raised_cos(t, w_in=omega[i]) if 1 <= t <= 3 else 0,
                                 np.arange(M1))])  # channel impulse response

    # construct the equalizer.
    f = FilterLMS(n=M2, mu=0.075, w='zeros')

    J = np.zeros((L, N))
    w = np.zeros((L, M2))
    for l in range(L):
        # generate the data.
        x = 2 * np.random.binomial(1, 0.5, N + M1 + M2 - 2) - 1

        # generate the noise.
        v = np.random.normal(scale=0.001, size=N + M2 - 1)

        # filter the data from the channel.
        u = fir_filter(input_from_history(x, M1), h) + v

        # calculate the equalizer output.
        d = x[delay:N + M1 + M2 - 1 - delay:]
        y, e, w[l] = f.run(d, input_from_history(u, M2))

        # calculate learning curve.
        J[l] = e ** 2

    J_avg = J.mean(axis=0)
    w_avg = w.mean(axis=0)
    ax1.semilogy(J_avg, label='$w = {}$'.format(omega[i]))
    ax2[i].stem(w_avg, label='$w = {}$'.format(omega[i]))

plt.grid(which='both')
plt.legend()
plt.show()
