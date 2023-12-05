import numpy as np
import itertools
from astropy.stats import sigma_clip
from .linear_model.linear_model import get_OLS_nd
from .utils import remove_bad, get_rank


def search_combination_OLS(X,
                           y,
                           n_components=2,
                           return_more=False,
                           is_sigma_clip=False,
                           sigma=3.0):
    """Search for the best combination of features using OLS.

    Args:
        x (np.ndarray): The features to use for training.
        y (np.ndarray): The labels to use for training.
        n_components (int, optional): The number of features to use.
            Defaults to 2.
        return_more (bool, optional): Whether to return all the results. Defaults
            to False.

    Returns:
        dict: The best combination of features and the corresponding OLS
            results.
    """
    X, y = remove_bad([X, y])

    if n_components > X.shape[1]:
        raise ValueError('n_components must be less than or equal to the '
                         'number of features.')

    results = {}

    for combination in itertools.combinations(range(X.shape[1]),
                                              n_components):
        if is_sigma_clip:
            _, _func = get_OLS_nd(X[:, combination], y)
            _diff = _func(X[:, combination]) - y
            bad = sigma_clip(_diff, sigma=sigma).mask
            results[combination] = get_OLS_nd(X[~bad][:, combination], y[~bad])
        else:
            results[combination] = get_OLS_nd(X[:, combination], y)

    mse_resid = np.array(
        [results[combination][0].mse_resid for combination in results])
    rank = get_rank(mse_resid)
    best_combination = list(results.keys())[np.argmin(mse_resid)]
    best_results = results[best_combination]

    if return_more:
        return best_combination, best_results, results, rank, mse_resid
    else:
        return best_combination, best_results
