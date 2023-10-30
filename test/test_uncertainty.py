import numpy as np
import asa.uncertainty as unc


class TestOneElement:

    def test_log(self):  # sourcery skip: bin-op-identity

        assert unc.log(1, 2) == np.abs(2 / 1)
        assert unc.log(2, 2) == np.abs(2 / 2)
        assert unc.log(3, 2) == np.abs(2 / 3)

    def test_log10(self):  # sourcery skip: bin-op-identity

        assert unc.log10(1, 2) == np.abs(2 / 1 / np.log(10))
        assert unc.log10(2, 2) == np.abs(2 / 2 / np.log(10))
        assert unc.log10(3, 2) == np.abs(2 / 3 / np.log(10))

    def test_log10_snr(self):  # sourcery skip: bin-op-identity
        assert unc.log10_snr(1, 2) == np.log10(1) / unc.log10(1, 2)
        assert unc.log10_snr(2, 2) == np.log10(2) / unc.log10(2, 2 / 2)
        assert unc.log10_snr(3, 2) == np.log10(3) / unc.log10(3, 3 / 2)

    def test_exp(self):

        assert unc.exp(1, 2) == np.abs(2 * np.exp(1))
        assert unc.exp(2, 2) == np.abs(2 * np.exp(2))
        assert unc.exp(3, 4) == np.abs(4 * np.exp(3))
        assert unc.exp(-3, 2) == np.abs(2 * np.exp(-3))

    def test_power(self):  # sourcery skip: bin-op-identity

        assert unc.power(1, 2, a=1) == np.abs(1 * 1**(1 - 1) * 2)
        assert unc.power(1, 2, a=2) == np.abs(2 * 1**(2 - 1) * 2)
        assert unc.power(2, 2, a=2) == np.abs(2 * 2**(2 - 1) * 2)
        assert unc.power(-2, 2, a=2) == np.abs(2 * (-2)**(2 - 1) * 2)
        assert unc.power(3, 4, a=-2) == np.abs(-2 * 3**(-2 - 1) * 4)

    def test_power_snr(self):

        # sourcery skip: no-loop-in-tests
        for _ in range(10):
            x = np.random.uniform(1, 10)
            x_snr = np.random.uniform(0.1, 10)
            a = np.random.uniform(-10, 10)
            assert np.isclose(unc.power_snr(x, x_snr, a=a), np.power(x, a) / unc.power(x, x/x_snr, a=a))
            # assert unc.power_snr(x, x_snr, a=a) == np.power(x, a) / unc.power(x, x/x_snr, a=a)

    def test_square(self):

        assert unc.square(1, 2) == np.abs(2 * 1**(2 - 1) * 2)
        assert unc.square(2, 2) == np.abs(2 * 2**(2 - 1) * 2)
        assert unc.square(3, 4) == np.abs(2 * 3**(2 - 1) * 4)
        assert unc.square(-2, 2) == np.abs(2 * (-2)**(2 - 1) * 2)

    def test_square_snr(self):

        # sourcery skip: no-loop-in-tests
        for _ in range(10):
            x = np.random.uniform(-10, 10)
            x_snr = np.random.uniform(0.1, 10)
            assert np.isclose(unc.square_snr(x, x_snr), np.square(x) / unc.square(x, x/x_snr))

class TestTwoElement:

    def test_sum(self):  # sourcery skip: bin-op-identity
        assert unc.sum([1, 2, 3]) == np.sqrt(1**2 + 2**2 + 3**2)

        assert unc.sum([1, 2, 3],
                       w=[1, 1,
                          1]) == np.sqrt((1 / 3)**2 + (2 / 3)**2 + (3 / 3)**2)
        assert unc.sum([1, 2, 3], w=[1, 1,
                                     0]) == np.sqrt((1 / 2)**2 + (2 / 2)**2)
        assert unc.sum([1, 2, 3],
                       w=[2, 2,
                          2]) == np.sqrt((1 / 3)**2 + (2 / 3)**2 + (3 / 3)**2)

        assert unc.sum([1, 2, 3], a=[1, 1, 1]) == np.sqrt(1**2 + 2**2 + 3**2)
        assert unc.sum([1, 2, 3], a=[1, 1, 0]) == np.sqrt(1**2 + 2**2)
        assert unc.sum([1, 2, 3],
                       a=[2, 2,
                          2]) == np.sqrt((1 * 2)**2 + (2 * 2)**2 + (3 * 2)**2)

    def test_ratio(self):

        assert unc.ratio(1, 2, 0.1,
                         0.2) == np.abs(1 / 2) * np.sqrt((0.1 / 1)**2 +
                                                         (0.2 / 2)**2)
        assert unc.ratio(6, 3, 1,
                         1) == np.abs(6 / 3) * np.sqrt((1 / 6)**2 + (1 / 3)**2)


class TestNElement:

    def test_multiply(self):

        assert unc.multiply(1, 1, 2, 2) == np.sqrt((1 * 2)**2 + (1 * 2)**2)
        assert unc.multiply(1, 2, 3, 4) == np.sqrt((1 * 4)**2 + (2 * 3)**2)


# class TestSNR:

#     def test_log10(self):  # sourcery skip: bin-op-identity

#         assert unc.log10(1, 2) == np.abs(2 / 1 / np.log(10))
#         assert unc.log10(2, 2) == np.abs(2 / 2 / np.log(10))
#         assert unc.log10(3, 2) == np.abs(2 / 3 / np.log(10))

#     def test_power(self):

#         assert unc.exp(1, 2) == np.abs(2 * np.exp(1))
#         assert unc.exp(2, 2) == np.abs(2 * np.exp(2))
#         assert unc.exp(3, 4) == np.abs(4 * np.exp(3))
#         assert unc.exp(-3, 2) == np.abs(2 * np.exp(-3))
