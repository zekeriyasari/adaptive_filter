from padapfilt.utils import *
from padapfilt.filters.base_filter import BaseFilter
import padapfilt.constants as co


class LMSFilter(BaseFilter):
    """
    LMS filter class.
    """

    _count = 0
    _kind = 'LMS'

    def __init__(self, m, w='zeros', mu=co.MU_LMS):
        super().__init__(m, w)
        self._mu = None
        self.mu = mu
        self._count += 1

    @property
    def mu(self):
        return self._mu

    @mu.setter
    def mu(self, value):
        try:
            value = float(value)
        except:
            raise TypeError('Step size cannot be converted to float')

        assert co.MU_LMS_MIN <= value <= co.MU_LMS_MAX, 'Step size must be in range({}, {})'.format(co.MU_LMS_MIN,
                                                                                                    co.MU_LMS_MAX)
        self._mu = value

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
        dw = self._mu * u * e
        self._w += dw
        return y, e

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
        e = np.zeros(n)
        for l in range(n):
            y[l], e[l] = self.adapt(d_vector[l], u_matrix[l])
        return y, e, self._w
