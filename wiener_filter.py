# Wiener filter implementation.


from utils import *
from scipy import signal
import matplotlib.pyplot as plt

N = 1024
M = 2
v1 = np.random.normal(scale=np.sqrt(0.27), size=N)
v2 = np.random.normal(scale=np.sqrt(1.0), size=N)

b1, a1 = np.array([1.0]), np.array([1, 0.8458])
b2, a2 = np.array([1.0]), np.array([1, -0.9458])
b3, a3 = np.array([1, 0.8360, -0.7853]), np.array([1.0])

d = signal.lfilter(b1, a1, v1)[len(b1) - 1:]
x = signal.lfilter(b2, a2, d)[len(b2) - 1:]
u = x + v2[:len(x)]
y = signal.lfilter(b3, a3, u)[len(a3) - 1:]

e = y - d
mse = np.mean(e * e)

fig, ax = plt.subplots(2)
ax[0].plot(d)
ax[1].plot(y)

ax[0].set_title('mes = {0}'.format(mse))
plt.show()
# TODO: Results are not as expected !!!
