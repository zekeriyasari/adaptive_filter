
from padapfilt.utils import *
from padapfilt.filters.base_filter import BaseFilter
import padapfilt.constants as co


class RLSFilter(BaseFilter):
    """
    RLS filter class.
    """
    _count = 0
    _kind = 'RLS'

    def __init__(self, m, delta = co.DELTA_RLS, w='random'):
        super().__init__(m, w)
        self._delta = None
        self.delta = delta
        self._count += 1

    @property
    def delta(self):
        return self._delta

    @delta.setter
    def delta(self, value):
        try:
            value = float(value)
        except:
            raise ValueError('Step size cannot be converted to float')

