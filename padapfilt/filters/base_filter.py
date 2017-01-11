

from padapfilt.utils import *
from padapfilt.constants import *


class BaseFilter(object):
    """
    Creates a base filter with `m` number of taps
    and `w` initial filter taps.
    """

    _count = 0

    def __init__(self, m, w='random'):
        self._m = None  # number of filter taps.
        self._w = None  # filter tap-weights.

        self.m = m
        self.w = w

        self._count += 1

    @property
    def m(self):
        return self._m

    @m.setter
    def m(self, value):
        assert type(value) == int and value > 0, 'Filter tap must be positive integer.'
        self._m = value

    @property
    def w(self):
        return self._w

    @w.setter
    def w(self, value):
        if type(value) == str and value in init_weight_opts.keys():
            self._w = np.random.normal(0, 0.5, self._m) if init_weight_opts[value] == 0 else np.zeros(self._m)
        elif type(value) == np.ndarray:
            assert len(value) == self._m, 'Length of {} is not to filter taps {}'.format(value, self._m)
            self._w = value
        else:
            raise TypeError('Cannot understand the {}'.format(value))

    def estimate(self, u):
        """
        Estimates the filter output for the input array
        :param u: ndarray,
            tap-input m-by-1 vector
        :return y: flaot
            output of the filter
        """

        assert type(u) == np.ndarray, 'Tap-input vector type must be np.array'
        assert len(u) == self._m, 'Length of tap-input vector {} is not equal to the' \
                                  'number of filter taps {}'.format(len(u), self._w)
        y = self._w.dot(u)
        return y



