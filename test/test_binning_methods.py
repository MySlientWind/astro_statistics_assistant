import numpy as np
from scipy.stats import binned_statistic, binned_statistic_2d

from asa.weighted_statistic import median, mean, std, std_mean, std_median, q
from asa.binning_methods import get_stat_method, bin_1d, binned_statistic_robust, binned_statistic_2d_robust
from asa.utils import flag_bad


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

    def test_binned_statistic_robust(self):

        # 1d
        x = np.random.normal(size=100)
        y = x + np.random.normal(size=100)

        statistic_res, bin_edges_res, binnumber_res = binned_statistic(
            x, y, statistic='mean', bins=10, range=(-3, 3))

        statistic_rb, bin_edges_rb, binnumber_rb = binned_statistic_robust(
            x, y, statistic='mean', bins=10, range=(-3, 3))

        assert np.array_equal(statistic_res, statistic_rb, equal_nan=True)
        assert np.array_equal(bin_edges_res, bin_edges_rb)
        assert np.array_equal(binnumber_res, binnumber_rb)

        _x = x.copy()
        _y = y.copy()

        _x[10:14] = np.nan
        _y[80:84] = np.inf

        is_bad = flag_bad(_x) | flag_bad(_y)

        statistic_rb, bin_edges_rb, binnumber_rb = binned_statistic_robust(
            _x, _y, statistic='mean', bins=10, range=(-3, 3))

        assert np.array_equal(binnumber_rb[~is_bad], binnumber_res[~is_bad])
        assert np.allclose(binnumber_rb[is_bad], -1)
        assert np.array_equal(bin_edges_res, bin_edges_rb)

        # 2d
        x = np.random.normal(size=100)
        y = np.random.normal(size=100)
        z = x**2 + y**2 + np.random.normal(size=100)

        statistic_res, x_edge_res, y_edge_res, binnumber_res = binned_statistic_2d(
            x,
            y,
            z,
            statistic='mean',
            bins=10,
            range=[(-3, 3), (-3, 3)],
            expand_binnumbers=True)

        statistic_rb, x_edge_rb, y_edge_rb, binnumber_rb = binned_statistic_2d_robust(
            x, y, z, statistic='mean', bins=10, range=[(-3, 3), (-3, 3)])

        assert np.array_equal(statistic_res, statistic_rb, equal_nan=True)
        assert np.array_equal(x_edge_res, x_edge_rb)
        assert np.array_equal(y_edge_res, y_edge_rb)
        assert np.array_equal(binnumber_res, binnumber_rb)

        _x = x.copy()
        _y = y.copy()
        _z = z.copy()

        _x[10:14] = np.nan
        _y[80:84] = np.inf
        _z[50:54] = np.nan

        is_bad = flag_bad(_x) | flag_bad(_y) | flag_bad(_z)

        statistic_rb, x_edge_rb, y_edge_rb, binnumber_rb = binned_statistic_2d_robust(
            _x, _y, _z, statistic='mean', bins=10, range=[(-3, 3), (-3, 3)])

        assert np.array_equal(binnumber_rb[0][~is_bad], binnumber_res[0][~is_bad])
        assert np.array_equal(binnumber_rb[1][~is_bad], binnumber_res[1][~is_bad])
        assert np.allclose(binnumber_rb[0][is_bad], -1)
        assert np.allclose(binnumber_rb[1][is_bad], -1)
        assert np.array_equal(x_edge_res, x_edge_rb)
        assert np.array_equal(y_edge_res, y_edge_rb)
