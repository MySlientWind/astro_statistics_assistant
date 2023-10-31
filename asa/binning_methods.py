from functools import partial

import numpy as np
from scipy.stats import binned_statistic, binned_statistic_2d

from .weighted_statistic import median, mean, std, std_mean, std_median, q

_range = range


def weighted_binned_statistic(x, y, w, bins=10, statistic=None, range=None):
    # TODO: least number in each bin
    _, edges, bin_index = binned_statistic(x,
                                           y,
                                           statistic='count',
                                           bins=bins,
                                           range=range)

    return np.array([
        statistic(y[bin_index == i], w[bin_index == i])
        for i in _range(1, len(edges))
    ])


def bin_2d(x, y, z, bins=10, range=None, min_data=0):
    # TODO: support with weights
    Z, x_edges, y_edges, _ = binned_statistic_2d(x,
                                                 y,
                                                 z,
                                                 statistic='mean',
                                                 bins=bins,
                                                 range=range)
    N, x_edges, y_edges, _ = binned_statistic_2d(x,
                                                 y,
                                                 z,
                                                 statistic='count',
                                                 bins=bins,
                                                 range=range)
    Z[N <= min_data] = np.nan
    Z = Z.T
    x_center, y_center = 0.5 * (x_edges[1:] + x_edges[:-1]), 0.5 * (
        y_edges[1:] + y_edges[:-1])
    X, Y = np.meshgrid(x_center, y_center)
    return X, Y, Z, x_edges, y_edges


def bin_1d(x,
           y,
           weights=None,
           x_statistic=None,
           y_statistic=None,
           bins=10,
           range=None,
           min_data=0):
    # TODO: count
    '''
    input:
        x_statistic, List[str]:
            'mean', 'median', 'std', 'std_mean', 'std_median', 'q:x' (x is a number between 0 and 1)
        y_statistic, List[str]:
            'mean', 'median', 'std', 'std_mean', 'std_median', 'q:x' (x is a number between 0 and 1)
    '''

    if x_statistic is None:
        x_statistic = []
    if y_statistic is None:
        y_statistic = ['mean']

    _, edges, bin_index = binned_statistic(x,
                                           y,
                                           statistic='count',
                                           bins=bins,
                                           range=range)

    # sourcery skip: dict-comprehension
    statistic = {}
    for x_stat in x_statistic:
        statistic[f'x_{x_stat}'] = []
    for y_stat in y_statistic:
        statistic[f'y_{y_stat}'] = []

    for i in _range(1, len(edges)):
        in_this_bin = (bin_index == i)
        N_in_this_bin = in_this_bin.sum()

        for x_stat in x_statistic:
            if N_in_this_bin <= min_data:
                statistic[f'x_{x_stat}'].append(np.nan)
            else:
                statistic[f'x_{x_stat}'].append(
                    get_stat_method(x_stat)(x[in_this_bin], weights[in_this_bin]))
        for y_stat in y_statistic:
            if N_in_this_bin <= min_data:
                statistic[f'y_{y_stat}'].append(np.nan)
            else:
                statistic[f'y_{y_stat}'].append(
                    get_stat_method(y_stat)(y[in_this_bin], weights[in_this_bin]))

    center = 0.5 * (edges[1:] + edges[:-1])

    return center, edges, bin_index, statistic


def get_stat_method(stat_name):
    mapper = {
        'mean': mean,
        'median': median,
        'std': partial(std, ddof=1),
        'std_mean': partial(std_mean, ddof=1),
        'std_median': partial(std_median, bandwidth='silverman'),
    }
    if stat_name.startswith('q:'):
        return partial(q, q=float(stat_name[2:]))
    else:
        return mapper[stat_name]
