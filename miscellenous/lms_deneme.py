
import numpy as np
import matplotlib.pyplot as plt


def lms(mu, M, u, d):

    w = np.zeros(M)
    N = len(u)

    e = d
    for n in range(N - M + 1):
        uvec = u[n:n + M][::-1]
        e[n] = d[n] - w.dot(uvec)
        w += mu * uvec * e[n]
    return e, w

h = np.array([0.5 ** i for i in range(5)])
u = np.random.randn(500)
d = np.convolve(h, u, 'valid')
e, w = lms(0.1, 5, u, d)
print(w)
