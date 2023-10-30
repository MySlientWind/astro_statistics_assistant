from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from .utils import flat_and_remove_bad

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
        rf = RandomForestClassifier(**RF_kwargs)
    elif problem_type == 'regression':
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

    if return_more:
        return feature_importance, rf, X_train, X_test, y_train, y_test
    else:
        return feature_importance


def get_correlation_coefficients(x, y):
    # TODO: more correlation coefficients
    x, y = flat_and_remove_bad([x, y])
    spearmanr_res = spearmanr(x, y)
    return {
        'spearmanr': (spearmanr_res[0], spearmanr_res[1]),
        'pearsonr': pearsonr(x, y)
    }