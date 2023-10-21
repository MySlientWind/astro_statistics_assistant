
import numpy as np
from asa.Bcorner import quantile

class TestQuantile:

    def test_quantile(self):
        x = np.random.normal(size=1000)
        assert np.isclose(quantile(x, 0.5)[0], np.median(x))
        assert quantile(list(x), 0.5)[0] == quantile(x, 0.5)[0]
        assert np.isnan(quantile([], 0.5)[0])
