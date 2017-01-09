
from padapfilt.utils import *
import padapfilt.constants as co


class LMSFilter(object):
    _count = 0

    def __init__(self, m, mu=co.MU_LMS, w='random'):
        self._kind = 'LMS'
        self._m = check_int(m, 'filter taps must be integers')
        self._mu = check_float_range(mu, co.MU_LMS_MIN, co.MU_LMS_MAX, 'mu')
        self._w = init_weights(w, self._m)
        self._count += 1

    def estimate(self, u):
        """
        compute filter output.

        :param u: ndarray,
            m-by-1 tap input vector
        :return y:  flaot,
            filter output i.e. estimated value of desired response.
        """

        y = self._w.dot(u)
        return y

    def adapt(self, d, u):
        """
        LMS filter adaptation.
        :param d: float,
            desired response.
        :param u: ndarray,
            m-by-1 tap input vector
        """
        y = self.estimate(u)
        e = d - y
        self._w += self._mu + u * e
        return y, e

    def run(self, d_vector, u_matrix):
        """
        Runs the filter for the data matrix and the desired vector d_vector.

        :param d_vector: ndarray,
            n-by-1 desired response.
        :param u_matrix: ndarray,
            n-by_n data matrix to be filtered.

        :return:
            y: ndarray
                n-by-1 filtered output
            w: ndarray
                m-by-1 filter tap weights
            e: ndarray
                n-by-1 filtering error.
        """

        n = u_matrix.shape[0]
        assert d_vector.size == n, 'The length of vector d and matrix x must agree.'
        assert type(u_matrix) == np.ndarray and type(d_vector) == np.ndarray, \
            'u_matrix and x_matrix must e numpy.ndarray'
        y = np.zeros(n)
        e = np.zeros(n)
        for l in range(n):
            y[l], e[l] = self.adapt(d_vector[l], u_matrix[l])
        return y, e, self._w



