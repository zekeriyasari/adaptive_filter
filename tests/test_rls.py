from unittest import TestCase
from padapfilt.filters.rls import *

import matplotlib.pyplot as plt


def raised_cos(x_in, w_in=2.9):
    """
    Raised cosine inter-symbol interference channel model.
    """
    return 0.5 * (1 + np.cos(2 * np.pi / w_in * (x_in - 2)))


class TestRLSFilter(TestCase):
    def test_rls(self):
        n = 1000
        m = 5
        x = np.random.normal(size=n + m - 1)  # input matrix
        h = np.array([1.0, 0.5, 0.25, 0.125, 0.0625])
        u = input_from_history(x, m)
        d = u.dot(h)

        f = RLSFilter(m, w='zeros', delta=0.004, lamda=0.98)
        y, ksi, w = f.run(d, u)

        print(w)
        # self.assertTrue(np.linalg.norm(eps) / np.linalg.norm(w_opt) < 0.1)
        # print('ok')



