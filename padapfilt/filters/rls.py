from padapfilt.utils import *
from padapfilt.filters.base_filter import BaseFilter
from padapfilt.constants import *


class RLSFilter(BaseFilter):
    """
    RLS filter class.
    """
    _count = 0
    _kind = 'RLS'

    def __init__(self, m, w='random', delta=DELTA_RLS, lamda=LAMDA_RLS):
        super().__init__(m, w)
        self._delta = delta
        self._lamda = lamda

        self.delta = delta
        self.lamda = lamda

        self._p_matrix = 1 / self._delta * np.identity(self._m)

        self._count += 1

    @property
    def delta(self):
        return self._delta

    @delta.setter
    def delta(self, value):
        try:
            value = float(value)
        except:
            raise TypeError('Step size cannot be converted to float')

        assert DELTA_RLS_MIN <= value <= DELTA_RLS_MAX, \
            'Regularization parameter must be in range({}, {})'.format(DELTA_RLS_MIN, DELTA_RLS_MAX)

        self._delta = value

    @property
    def lamda(self):
        return self._delta

    @lamda.setter
    def lamda(self, value):
        try:
            value = float(value)
        except:
            raise TypeError('Step size cannot be converted to float')

        assert LAMDA_RLS_MIN <= value <= LAMDA_RLS_MAX, \
            'Regularization parameter must be in range({}, {})'.format(LAMDA_RLS_MIN, LAMDA_RLS_MAX)

        self._lamda = value

    def adapt(self, d, u):
        """
        LMS filter adaptation.
        :param d: float,
            desired response.
        :param u: ndarray,
            m-by-1 tap input vector
        """

        num = self._p_matrix.dot(u)
        den = self._lamda + u.dot(self._p_matrix.dot(u))
        k = num / den
        y = self.estimate(u)
        ksi = d - y
        dw = k * ksi
        self._w += dw
        self._p_matrix = 1 / self._lamda * (self._p_matrix - np.outer(k, u.dot(self._p_matrix)))
        return y, ksi

    def run(self, d_vector, u_matrix):
        """
        Runs the filter for the data matrix and the desired vector d_vector.

        :param d_vector: ndarray,
            n-by-1 desired response.
        :param u_matrix: ndarray,
            n-by_n data matrix to be filtered.

        :return:
            y_a: ndarray
                n-by-1 filtered output
            w_a: ndarray
                m-by-1 filter tap weights
            ksi: ndarray
                n-by-1 filtering error.
        """

        n = u_matrix.shape[0]
        assert d_vector.size == n, 'The length of vector d and matrix x must agree.'
        assert type(u_matrix) == np.ndarray and type(d_vector) == np.ndarray, \
            'u_matrix and x_matrix must ksi numpy.ndarray'

        y = np.zeros(n)
        ksi = np.zeros(n)
        for l in range(1, n):
            y[l], ksi[l] = self.adapt(d_vector[l], u_matrix[l])
        return y, ksi, self._w
