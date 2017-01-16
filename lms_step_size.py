import numpy as np
from scipy.linalg import toeplitz

# TODO: Correlation R matrix will be written.


def raised_cos(x_in, w_in=3.5):
    """
    Raised cosine inter-symbol interference channel model.
    """
    return 0.5 * (1 + np.cos(2 * np.pi / w_in * (x_in - 2)))


m = 11

var = 0.001
# h = np.array([raised_cos(x) for x in range(1, 4)])
h = np.array([0.25, 1, 0.25])

r0 = h[0] ** 2 + h[1] ** 2 + h[2] ** 2 + var
r1 = h[0] * h[1] + h[1] * h[2]
r2 = h[0] * h[2]
c = np.append(np.array([r0, r1, r2]), np.zeros(m - len(h)))
R = toeplitz(c)
lambdas = np.linalg.eigvals(R)
print(2 / m / r0)
