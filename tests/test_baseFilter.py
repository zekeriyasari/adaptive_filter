from unittest import TestCase
from padapfilt.filters.base_filter import *
from padapfilt.utils import *


class TestBaseFilter(TestCase):
    def test_base_filter(self):
        m = 3
        f1 = BaseFilter(m, w='zeros')
        self.assertTrue(np.allclose(f1._w, np.zeros(m)))
        print('Ok. Initialized filter with zero taps.')

        w0 = np.array([0.1, 0.2, 0.3])
        f1 = BaseFilter(m, w=w0)
        self.assertTrue(np.allclose(f1._w, w0))
        print('Ok. Initialized filter with specified taps.')

        u = np.array([1.0, 2.0, 3.0])
        y = f1.estimate(u)
        self.assertTrue(np.allclose(y, np.array([1.4])))
        print('Ok. Filter estimation passed..')

        print('Ok. Initialization of the filter.')













