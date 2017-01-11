from unittest import TestCase
from padapfilt.utils import *


class TestUtils(TestCase):
    """
    test case for padapfilt.utils module.
    """

    def test_input_from_history(self):
        m = 3
        data = np.arange(5)
        u_matrix = input_from_history(data, m)
        self.assertTrue(np.allclose(u_matrix, np.array([[2, 1, 0],
                                                        [3, 2, 1],
                                                        [4, 3, 2]]))), 'failed'
        print('passed...')

    def test_acf(self):
        pass
