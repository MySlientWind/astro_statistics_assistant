import numpy as np

from asa.feature_selection_methods import search_combination_OLS
from asa.linear_model import get_OLS_nd

class TestFunc:
    def test_search_combination_OLS(self):
        X = np.random.rand(100, 10)
        y = X[:, 2] + 0.5 * X[:, 5] + 0.1 * X[:, 8] +  np.random.rand(100)
        best_combination, best_results, _results, rank, res_metric = search_combination_OLS(X, y, return_more=True)
        assert best_combination == (2, 5)
        results, func = get_OLS_nd(X[:,[2, 5]], y)
        assert np.allclose(best_results[0].params, results.params)
        assert np.allclose(best_results[1](X[:, [2, 5]]), func(X[:, [2, 5]]))
        assert res_metric[np.argmin(rank)] == best_results[0].mse_resid
        assert res_metric[np.argmin(rank)] == results.mse_resid

        best_combination, best_results, _results, rank, res_metric = search_combination_OLS(X, y, return_more=True, metric='r2')
        assert res_metric[np.argmin(rank)] == best_results[0].rsquared
        assert res_metric[np.argmin(rank)] == results.rsquared

        best_combination, best_results, _results, rank, res_metric = search_combination_OLS(X, y, return_more=True, metric='bic')
        assert res_metric[np.argmin(rank)] == best_results[0].bic
        assert res_metric[np.argmin(rank)] == results.bic

        X = np.random.rand(100, 10)
        y = X[:, 2] + 0.5 * X[:, 5] + 0.1 * X[:, 8] +  np.random.rand(100)
        y[10:20] = 10
        best_combination, best_results = search_combination_OLS(X, y, is_sigma_clip=True)
        assert best_combination == (2, 5)

