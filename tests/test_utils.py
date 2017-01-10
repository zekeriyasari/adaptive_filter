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
        self.assertTrue(np.allclose(u_matrix, np.array([[0, 1, 2],
                                                        [1, 2, 3],
                                                        [2, 3, 4]]))), 'failed'
        print('passed...')

    def test_acf(self):
        pass

    def test_fir_filter(self):
        pass

    def test_init_weights(self):
        pass




