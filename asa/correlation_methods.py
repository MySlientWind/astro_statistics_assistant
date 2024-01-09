import numpy as np
from scipy.stats import pearsonr, spearmanr

try:
    from sklearnex.ensemble import RandomForestClassifier as RandomForestClassifier_ex
    from sklearnex.ensemble import RandomForestRegressor as RandomForestRegressor_ex
    EX_AVAILABLE = True
except ImportError:
    EX_AVAILABLE = False

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from scipy.special import digamma
from math import log
import scipy.spatial as ss
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from scipy.optimize import differential_evolution

from .utils import remove_bad

USE_EX = True

def get_RF_importance(x,
                      y,
                      problem_type,
                      importance_type='gini',
                      test_size=0.2,
                      return_more=False,
                      RF_kwargs=None):
    """Get the importance of each feature in a dataset using a random forest.

    Args:
        x (np.ndarray): The data to use for training the random forest.
        y (np.ndarray): The labels to use for training the random forest.
        problem_type (str): The type of problem to solve. Either 'classification'
            or 'regression'.
        importance_type (str): The type of importance to calculate. Either
            'gini' or 'permutation'.

    Returns:
        np.ndarray: The importance of each feature in the dataset.
    """

    if RF_kwargs is None:
        RF_kwargs = {}

    X_train, X_test, y_train, y_test = train_test_split(x,
                                                        y,
                                                        test_size=test_size)

    if problem_type == 'classification':
        if EX_AVAILABLE and USE_EX:
            rf = RandomForestClassifier_ex(**RF_kwargs)
        else:
            rf = RandomForestClassifier(**RF_kwargs)
    elif problem_type == 'regression':
        if EX_AVAILABLE and USE_EX:
            rf = RandomForestRegressor_ex(**RF_kwargs)
        else:
            rf = RandomForestRegressor(**RF_kwargs)
    else:
        raise ValueError('problem_type must be either classification or '
                         'regression.')

    rf.fit(X_train, y_train)

    if importance_type == 'gini':
        feature_importance = rf.feature_importances_
    elif importance_type == 'permutation':
        feature_importance = permutation_importance(rf,
                                                    X_test,
                                                    y_test,
                                                    n_repeats=10)

    score_train = rf.score(X_train, y_train)
    score_test = rf.score(X_test, y_test)

    if return_more:
        return feature_importance, score_test, score_train, rf, X_train, X_test, y_train, y_test
    else:
        return feature_importance, score_test


def get_correlation_coefficients(x, y):
    # TODO: more correlation coefficients
    x, y = remove_bad([x, y])
    spearmanr_res = spearmanr(x, y)
    return {
        'spearmanr': (spearmanr_res[0], spearmanr_res[1]),
        'pearsonr': pearsonr(x, y)
    }


def get_MI(x,
           y,
           k=5,
           is_qt=False,
           x_qt_kwargs=None,
           y_qt_kwargs=None,
           robust=False,
           x_scaler_bounds=None,
           y_scaler_bounds=None,
           da_kwargs=None):
    # remove bad
    x, y = remove_bad([x, y])

    # at least 2D
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)

    # TODO: other preprocessing
    # seems qt is not good
    if is_qt:
        if x_qt_kwargs is None:
            x_qt_kwargs = {}
        if y_qt_kwargs is None:
            y_qt_kwargs = {}
        x_qt = QuantileTransformer(**x_qt_kwargs)
        y_qt = QuantileTransformer(**y_qt_kwargs)
        _x = x_qt.fit_transform(x)
        _y = y_qt.fit_transform(y)
    else:
        _x = x
        _y = y

    # here we use DE to optimize the scaler, we choose the max MI
    if robust:
        n_x = x.shape[1]
        n_y = y.shape[1]

        def target(scaler):
            x_scaler = scaler[:n_x]
            y_scaler = scaler[n_x:]
            return -kraskov_mi(x_scaler * _x, y_scaler * _y, k=k)

        if x_scaler_bounds is None:
            x_scaler_bounds = [(0.5, 2)] * n_x
        if y_scaler_bounds is None:
            y_scaler_bounds = [(0.5, 2)] * n_y

        bounds = x_scaler_bounds + y_scaler_bounds

        if da_kwargs is None:
            da_kwargs = {}

        res = differential_evolution(target, bounds, **da_kwargs)

        x_scaler = res.x[:n_x]
        y_scaler = res.x[n_x:]

        return x_scaler, y_scaler, kraskov_mi(x_scaler * _x,
                                              y_scaler * _y,
                                              k=k)

    else:
        return kraskov_mi(_x, _y, k=k)


def kraskov_mi(x, y, k=5):
    '''
        Estimate the mutual information I(X;Y) of X and Y from samples {x_i, y_i}_{i=1}^N
        Using KSG mutual information estimator

        Input: x: 2D list of size N*d_x
        y: 2D list of size N*d_y
        k: k-nearest neighbor parameter

        Output: one number of I(X;Y)
    '''

    assert len(x) == len(y), "Lists should have same length"
    assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
    N = len(x)
    dx = len(x[0])
    dy = len(y[0])
    data = np.concatenate((x, y), axis=1)

    tree_xy = ss.cKDTree(data)
    tree_x = ss.cKDTree(x)
    tree_y = ss.cKDTree(y)

    knn_dis = [
        tree_xy.query(point, k + 1, p=float('inf'))[0][k] for point in data
    ]
    ans_xy = -digamma(k) + digamma(N) + (dx + dy) * log(
        2)  #2*log(N-1) - digamma(N) #+ vd(dx) + vd(dy) - vd(dx+dy)
    ans_x = digamma(N) + dx * log(2)
    ans_y = digamma(N) + dy * log(2)
    for i in range(N):
        ans_xy += (dx + dy) * log(knn_dis[i]) / N
        ans_x += -digamma(
            len(
                tree_x.query_ball_point(
                    x[i], knn_dis[i] - 1e-15, p=float('inf')))) / N + dx * log(
                        knn_dis[i]) / N
        ans_y += -digamma(
            len(
                tree_y.query_ball_point(
                    y[i], knn_dis[i] - 1e-15, p=float('inf')))) / N + dy * log(
                        knn_dis[i]) / N

    return ans_x + ans_y - ans_xy
