import numpy as np
from statsmodels.stats.weightstats import DescrStatsW

from asa.weighted_statistic import *

class TestWeightedStatistic:

    def test_effect_sample_size(self):
        w = np.ones(100)
        assert get_effect_sample_size(w) == 100

    def test_shape(self):
        x = np.random.normal(size=100)
        w = np.random.uniform(size=100)

        assert median(x, w).shape == ()
        assert median(x).shape == ()

        assert mean(x, w).shape == ()
        assert mean(x).shape == ()

        assert std(x, w).shape == ()
        assert std(x).shape == ()

        assert std_mean(x, w).shape == ()
        assert std_mean(x).shape == ()

        assert std_median(x, w).shape == ()
        assert std_median(x).shape == ()

        assert q(x, w).shape == ()
        assert q(x).shape == ()

    def test_res(self):
        x = np.random.normal(size=100)
        w = np.random.uniform(size=100)

        assert mean(x) == np.mean(x)
        assert median(x) == np.median(x)
        assert std(x) == np.std(x)
        assert std_mean(x) == np.std(x) / np.sqrt(len(x))

        ds = DescrStatsW(x, weights=w)
        assert np.isclose(mean(x, w), ds.mean)


        assert np.isclose(std(x, ddof=1), np.std(x, ddof=1))
        assert np.isclose(std(x), std(x, w=np.ones_like(x)))
        assert np.isclose(std(x, ddof=1), std(x, w=np.ones_like(x), ddof=1))
        assert np.isclose(std(x, ddof=1), std(x, w=np.ones_like(x)*2, ddof=1))

        assert np.isclose(std_mean(x), std_mean(x, w=np.ones_like(x)))
        assert np.isclose(std_mean(x, ddof=1), std_mean(x, w=np.ones_like(x), ddof=1))
        assert np.isclose(std_mean(x, ddof=1), std_mean(x, w=np.ones_like(x)*2, ddof=1))

        assert np.isclose(std_median(x), std_median(x, w=np.ones_like(x)))
        assert np.isclose(std_median(x), std_median(x, w=np.ones_like(x) * 2))


