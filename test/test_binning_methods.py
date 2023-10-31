import numpy as np

from asa.weighted_statistic import median, mean, std, std_mean, std_median, q
from asa.binning_methods import get_stat_method, bin_1d


class TestGetStatMethod:

    def test_mean(self):
        x = np.random.normal(size=100)
        w = np.random.uniform(size=100)
        assert get_stat_method('mean')(x, w) == mean(x, w)

    def test_median(self):
        x = np.random.normal(size=100)
        w = np.random.uniform(size=100)
        assert get_stat_method('median')(x, w) == median(x, w)

    def test_std(self):
        x = np.random.normal(size=100)
        w = np.random.uniform(size=100)
        assert get_stat_method('std')(x, w) == std(x, w, ddof=1)

    def test_std_mean(self):
        x = np.random.normal(size=100)
        w = np.random.uniform(size=100)
        assert get_stat_method('std_mean')(x, w) == std_mean(x, w, ddof=1)

    def test_std_median(self):
        x = np.random.normal(size=100)
        w = np.random.uniform(size=100)
        assert get_stat_method('std_median')(x, w) == std_median(
            x, w, bandwidth='silverman')

    def test_q(self):
        x = np.random.normal(size=100)
        w = np.random.uniform(size=100)
        assert get_stat_method('q:0.1')(x, w) == q(x, w, q=0.1)
        assert get_stat_method('q:0.5')(x, w) == q(x, w, q=0.5)
        assert get_stat_method('q:0.9')(x, w) == q(x, w, q=0.9)


class TestBinMethod:

    def test_bin1d(self):
        x = np.random.normal(size=100)
        y = 3 * x + np.random.normal(size=100)
        w = np.random.uniform(size=100)

        center, edges, bin_index, statistic = bin_1d(x,
                                                     y,
                                                     weights=w,
                                                     x_statistic=[
                                                         'mean', 'median',
                                                         'std', 'std_mean',
                                                         'std_median', 'q:0.3',
                                                         'q:0.7'
                                                     ],
                                                     y_statistic=[
                                                         'mean', 'median',
                                                         'std', 'std_mean',
                                                         'std_median', 'q:0.3',
                                                         'q:0.7'
                                                     ],
                                                     range=(-3, 3))

        edges_true = np.linspace(-3, 3, 11)

        assert np.array_equal(center, 0.5 * (edges_true[1:] + edges_true[:-1]))
        assert np.array_equal(edges, edges_true)

        assert np.array_equal(bin_index, np.digitize(x, edges))

        is_bin_6 = bin_index == 6
        assert np.array_equal(
            statistic['x_mean'][5],
            get_stat_method('mean')(x[is_bin_6], w[is_bin_6]))
        assert np.array_equal(
            statistic['x_median'][5],
            get_stat_method('median')(x[is_bin_6], w[is_bin_6]))
        assert np.array_equal(statistic['x_std'][5],
                              get_stat_method('std')(x[is_bin_6], w[is_bin_6]))
        assert np.array_equal(
            statistic['x_std_mean'][5],
            get_stat_method('std_mean')(x[is_bin_6], w[is_bin_6]))

        assert np.array_equal(np.isnan(statistic['x_median']),
                              np.isnan(statistic['x_std_median']))
        assert np.array_equal(np.isnan(statistic['y_median']),
                              np.isnan(statistic['y_std_median']))
