import numpy as np
import itertools
from astropy.stats import sigma_clip
from .linear_model.linear_model import get_OLS_nd
from .utils import remove_bad, get_rank
from .sklearn_tools.auto_tune import get_RF_CVS


def search_combination_OLS(X,
                           y,
                           n_components=2,
                           allowe_small_n=False,
                           return_more=False,
                           metric='mse_resid',
                           is_sigma_clip=False,
                           sigma=3.0):
    """Search for the best combination of features using OLS.

    Args:
        x (np.ndarray): The features to use for training.
        y (np.ndarray): The labels to use for training.
        n_components (int, optional): The number of features to use.
            Defaults to 2.
        allowe_small_n (bool, optional): Whether to allow the number of
            features to be less than n_components. Defaults to False.
        return_more (bool, optional): Whether to return all the results. Defaults
            to False.
        metric (str, optional): The metric to use for selecting the best
            combination. Defaults to 'mse_resid'.
        is_sigma_clip (bool, optional): Whether to use sigma clipping to
            remove outliers. Defaults to False.
        sigma (float, optional): The sigma value to use for sigma clipping.
            Defaults to 3.0.

    Returns:
        dict: The best combination of features and the corresponding OLS
            results.
    """
    def get_metric(results, metric='mse_resid'):
        if metric == 'mse_resid':
            return results.mse_resid
        elif metric == 'r2':
            return results.rsquared
        elif metric == 'bic':
            return results.bic

    if metric in ['mse_resid', 'bic']:
        rank_scaler = 1
    elif metric == 'r2':
        rank_scaler = -1

    X, y = remove_bad([X, y])

    if n_components > X.shape[1]:
        raise ValueError('n_components must be less than or equal to the '
                         'number of features.')

    results = {}

    if allowe_small_n:
        all_combinations = itertools.chain(*[
            itertools.combinations(range(X.shape[1]), n_components - i)
            for i in range(n_components)
        ])
    else:
        all_combinations = itertools.combinations(range(X.shape[1]),
                                                  n_components)

    for combination in all_combinations:
        if is_sigma_clip:
            _, _func = get_OLS_nd(X[:, combination], y)
            _diff = _func(X[:, combination]) - y
            bad = sigma_clip(_diff, sigma=sigma).mask
            results[combination] = get_OLS_nd(X[~bad][:, combination], y[~bad])
        else:
            results[combination] = get_OLS_nd(X[:, combination], y)

    res_metric = np.array([
        get_metric(results[combination][0], metric=metric)
        for combination in results
    ])
    rank = get_rank(rank_scaler * res_metric)
    best_combination = list(results.keys())[np.argmin(rank)]
    best_results = results[best_combination]

    if return_more:
        return best_combination, best_results, results, rank, res_metric
    else:
        return best_combination, best_results


def search_combination_RF_cls(X,
                              y,
                              n_components=2,
                              allowe_small_n=False,
                              return_more=False,
                              metric='balanced_accuracy',
                              CVS_method='grid',
                              param_grid='basic',
                              param_distributions=None,
                              CVS_kwargs=None):
    """Search for the best combination of features using RF.

    Args:
        x (np.ndarray): The features to use for training.
        y (np.ndarray): The labels to use for training.
        n_components (int, optional): The number of features to use.
            Defaults to 2.
        allowe_small_n (bool, optional): Whether to allow the number of
            features to be less than n_components. Defaults to False.
        return_more (bool, optional): Whether to return all the results. Defaults
            to False.
        metric (str, optional): The metric to use for selecting the best
            combination. Defaults to 'balanced_accuracy'.

    Returns:
        dict: The best combination of features and the corresponding OLS
            results.
    """

    if CVS_kwargs is None:
        CVS_kwargs = {}

    elif metric in ['balanced_accuracy', 'accuracy']:
        CVS_kwargs['scoring'] = metric
        rank_scaler = -1

    X, y = remove_bad([X, y])

    if n_components > X.shape[1]:
        raise ValueError('n_components must be less than or equal to the '
                         'number of features.')

    results = {}

    if allowe_small_n:
        all_combinations = itertools.chain(*[
            itertools.combinations(range(X.shape[1]), n_components - i)
            for i in range(n_components)
        ])
    else:
        all_combinations = itertools.combinations(range(X.shape[1]),
                                                  n_components)

    for combination in all_combinations:
        results[combination] = get_RF_CVS(
            X[:, combination],
            y,
            'classification',
            CVS_method=CVS_method,
            param_grid=param_grid,
            param_distributions=param_distributions,
            CVS_kwargs=CVS_kwargs)

    res_metric = np.array([
        -rank_scaler * results[combination].best_score_
        for combination in results
    ])
    rank = get_rank(rank_scaler * res_metric)
    best_combination = list(results.keys())[np.argmin(rank)]
    best_results = results[best_combination]

    if return_more:
        return best_combination, best_results, results, rank, res_metric
    else:
        return best_combination, best_results


def search_combination_RF_reg(X,
                              y,
                              n_components=2,
                              allowe_small_n=False,
                              return_more=False,
                              metric='mse_resid',
                              CVS_method='grid',
                              param_grid='basic',
                              param_distributions=None,
                              CVS_kwargs=None):
    """Search for the best combination of features using RF.

    Args:
        x (np.ndarray): The features to use for training.
        y (np.ndarray): The labels to use for training.
        n_components (int, optional): The number of features to use.
            Defaults to 2.
        allowe_small_n (bool, optional): Whether to allow the number of
            features to be less than n_components. Defaults to False.
        return_more (bool, optional): Whether to return all the results. Defaults
            to False.
        metric (str, optional): The metric to use for selecting the best
            combination. Defaults to 'mse_resid'.

    Returns:
        dict: The best combination of features and the corresponding OLS
            results.
    """

    if CVS_kwargs is None:
        CVS_kwargs = {}

    if metric == 'mse_resid':
        CVS_kwargs['scoring'] = 'neg_mean_squared_error'
        rank_scaler = 1

    elif metric == 'r2':
        CVS_kwargs['scoring'] = 'r2'
        rank_scaler = -1

    X, y = remove_bad([X, y])

    if n_components > X.shape[1]:
        raise ValueError('n_components must be less than or equal to the '
                         'number of features.')

    results = {}

    if allowe_small_n:
        all_combinations = itertools.chain(*[
            itertools.combinations(range(X.shape[1]), n_components - i)
            for i in range(n_components)
        ])
    else:
        all_combinations = itertools.combinations(range(X.shape[1]),
                                                  n_components)

    for combination in all_combinations:
        results[combination] = get_RF_CVS(
            X[:, combination],
            y,
            'regression',
            CVS_method=CVS_method,
            param_grid=param_grid,
            param_distributions=param_distributions,
            CVS_kwargs=CVS_kwargs)

    res_metric = np.array([
        -rank_scaler * results[combination].best_score_
        for combination in results
    ])
    rank = get_rank(rank_scaler * res_metric)
    best_combination = list(results.keys())[np.argmin(rank)]
    best_results = results[best_combination]

    if return_more:
        return best_combination, best_results, results, rank, res_metric
    else:
        return best_combination, best_results
