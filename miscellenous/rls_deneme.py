
import numpy as np
import matplotlib.pyplot as plt


def rls(lamda, M, u, d, delta):

    # TODO: Outer product
    w = np.zeros(M)
    P = np.eye(M) / delta

    N = len(u)

    xi = np.zeros_like(u)
    for n in range(N - M + 1):
        uvec = u[n:n + M][::-1]  # must be a column vector
        lamda_inv = lamda ** -1
        k = lamda_inv * P.dot(uvec) / (1 + lamda_inv * uvec.dot(P.dot(uvec)))
        xi[n] = d[n] - w.dot(uvec)
        w += k * xi[n]
        P = lamda_inv * P - lamda_inv * np.outer(k, uvec.dot(P))
    return xi, w

h = np.array([0.5 ** i for i in range(5)])
u = np.random.randn(1000)
d = np.convolve(h, u, 'valid')
e, w = rls(0.98, 5, u, d, 0.004)
print(w)
