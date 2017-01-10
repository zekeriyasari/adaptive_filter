import numpy as np


def rls(lamda, M, u, d, delta):
    w = np.zeros(M)
    P = np.eye(M)/delta

    N = len(u)

    xi = d

    for n in range(N-M):
        uvec = u[n: n+M:][::-1]
        k = (lamda ** -1) * np.dot(P, uvec) / (1 + (lamda ** -1) * uvec.T.dot(P.dot(uvec)))
        xi[n] = d[n] - w.dot(uvec)
        w += k*xi[n]
        P = (lamda ** -1) * P - (lamda ** -1) * k.dot(uvec.T.dot(P))
    return xi, w

h = np.array([1.0,  0.5, 0.25, 0.125, 0.0625])
u = np.random.normal(size=1000)
d = np.convolve(u, h, 'full')
xi, w = rls(1, 5, u, d, 0.005)
print(w)
