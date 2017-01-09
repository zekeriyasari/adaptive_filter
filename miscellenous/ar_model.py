# AR stochastic process.

from scipy import signal
import matplotlib.pyplot as plt

from miscellenous.utils import *

N = 256  # number of data points.
M = 2  # number of filter taps.
L = 100  # number of auto-correlation lags

# generate white noise.
v = np.random.normal(scale=np.sqrt(0.0731), size=N)

# construct the IIR filter.
b = np.array([1.0])
a = np.array([1, -0.975, 0.95])

# compute the
u = signal.lfilter(b, a, v)[M-1:]

# compute auto-correlation function.
r = np.zeros(L)
for l in range(L):
    r[l] = acf(u, l)

fig, ax = plt.subplots(3)
ax[0].plot(v)
ax[1].plot(u)
ax[2].stem(r/r[0])

plt.show()
