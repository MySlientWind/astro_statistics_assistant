from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


def get_RF_CVS(x,
               y,
               problem_type,
               CVS_method='grid',
               param_grid='basic',
               param_distributions=None,
               CVS_kwargs=None):

    if CVS_kwargs is None:
        CVS_kwargs = {}

    if problem_type == 'classification':
        rf = RandomForestClassifier()
    elif problem_type == 'regression':
        rf = RandomForestRegressor()

    # predined param_grid
    if param_grid == 'basic':
        param_grid = {
            'n_estimators': [10, 100, 1000],
            'max_depth': [5, 11, 31, 51, 101]
        }

    if CVS_method == 'grid':
        if param_grid is None:
            raise ValueError('param_grid must be provided if CVS_method is '
                             '"grid".')
        cvs = GridSearchCV(rf, param_grid, **CVS_kwargs)
    elif CVS_method == 'random':
        if param_distributions is None:
            raise ValueError('param_distributions must be provided if '
                             'CVS_method is "random".')
        cvs = RandomizedSearchCV(rf, param_distributions, **CVS_kwargs)

    cvs.fit(x, y)

    return cvs
