from unittest import TestCase
from padapfilt.filters.lms import *

import matplotlib.pyplot as plt


def raised_cos(x_in, w_in=2.9):
    """
    Raised cosine inter-symbol interference channel model.
    """
    return 0.5 * (1 + np.cos(2 * np.pi / w_in * (x_in - 2)))


class TestLMSFilter(TestCase):
    def test_lms(self):
        n = 500
        m = 4
        x = np.random.normal(0, 1, (n, m))  # input matrix
        w_opt = np.array([2.0, 0.1, -4.0, 0.5])
        d = x.dot(w_opt) + np.random.normal(0, 0.1, n)

        f = LMSFilter(m, mu=0.1, w='zeros')
        y, e, w = f.run(d, x)

        eps = w - w_opt
        self.assertTrue(np.linalg.norm(eps) / np.linalg.norm(w_opt) < 0.1)
        print('ok')






