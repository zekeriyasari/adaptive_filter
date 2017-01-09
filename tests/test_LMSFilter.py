from unittest import TestCase
from padapfilt.filters.lms import *
import matplotlib.pyplot as plt


class TestLMSFilter(TestCase):
    def test_estimate(self):
        pass

    def test_adapt(self):
        pass

    def test_run(self):
        # determine simulation parameters.
        n = 100  # number of input data samples to the equalizer.
        m1 = 4  # number of taps of channel.
        m2 = 11  # number of taps of equalizer
        l = 11  # number of trials.
        delay = int(m1 / 2) + int(m2 / 2)

        omega = np.array([2.9])

        for i in range(len(omega)):
            # construct the channel.
            h = np.array([y for y in map(lambda t: raised_cos(t, w_in=omega[i]) if 1 <= t <= 3 else 0,
                                         np.arange(m1))])  # channel impulse response

            # construct the equalizer.
            f = LMSFilter(m=m2, mu=0.075, w='zeros')

            j = np.zeros((l, n))
            w = np.zeros((l, m2))
            for l in range(l):
                # generate the data.
                x = 2 * np.random.binomial(1, 0.5, n + m1 + m2 - 2) - 1

                # generate the noise.
                v = np.random.normal(scale=0.001, size=n + m2 - 1)

                # filter the data from the channel.
                u = fir_filter(input_from_history(x, m1), h) + v

                # calculate the equalizer output.
                d = x[delay:n + m1 + m2 - 1 - delay:]
                y, e, w[l] = f.run(d, input_from_history(u, m2))

                # calculate learning curve.
                j[l] = e ** 2
            j_avg = j.mean(axis=0)
            plt.plot(j_avg)
            plt.show()
