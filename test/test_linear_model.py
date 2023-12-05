import numpy as np
from asa.linear_model import get_OLS_nd


class TestLinearModel:
    def test_get_OLS_nd(self):
        X = np.random.normal(size=(1000, 3))
        y = X @ np.array([1, 0.1, -1]) + np.random.normal(size=1000) + 0.5
        results, func = get_OLS_nd(X, y)

        assert results.params.shape == (4, )
        assert func(X).shape == (1000, )
        assert np.allclose(results.params,
                           np.array([0.5, 1, 0.1, -1]),
                           rtol=0.1,
                           atol=0.05)
