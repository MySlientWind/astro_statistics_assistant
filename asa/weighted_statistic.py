import numpy as np
from sklearn.neighbors import KernelDensity

from .Bcorner import quantile as _quantile

# TODO: in batch by axis

def quantile(x, q, weights=None, N_min=2):
    '''
    The defination of weighted quantile is different from DescrStatsW
    We use the interpolation of cdf, but they use the ecdf
    '''
    return np.array(_quantile(x, q, weights, N_min)[0])


def median(x, w=None):
    return np.median(x) if w is None else quantile(x, 0.5, weights=w)


def mean(x, w=None):
    return np.mean(x) if w is None else np.average(x, weights=w)


def std(x, w=None, ddof=0):  # sourcery skip: remove-unnecessary-else
    '''
    Note that the w here is considered as reliability weights.
    However, the weights in DescrStatsW is frequency weights.

    https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_variance
    
    Because the ddof with weight is a so complex concept
    We use the effect sample size for ddof correction in weighted std
    https://en.wikipedia.org/wiki/Design_effect#Effective_sample_size
    '''
    N = get_effect_sample_size(w) if w is not None else x.size
    if N - ddof <= 0:
        return np.nan
    if w is None:
        return np.std(x, ddof=ddof)
    else:
        return np.sqrt(
            np.average(
                (x - mean(x, w))**2, weights=w) * N / (N - ddof))


def std_mean(x, w=None, ddof=0):
    '''
    The ddof only affects the estimation of std
    We use the effect sample size as the denominator
    https://en.wikipedia.org/wiki/Design_effect#Effective_sample_size
    '''
    N = get_effect_sample_size(w) if w is not None else x.size
    return std(x, w=w, ddof=ddof) / np.sqrt(N)


def std_median(x, w=None, **kde_kwargs):
    kde = KernelDensity(**kde_kwargs).fit(x.reshape(-1, 1), sample_weight=w)
    m = median(x, w)
    if np.isnan(m):
        return np.nan
    fm = np.exp(kde.score_samples(np.array([[m]])))
    N = get_effect_sample_size(w) if w is not None else x.size
    return 1 / (2 * fm * np.sqrt(N))[0]


def q(x, w=None, q=0.5):
    return quantile(x, q, weights=w)


def get_effect_sample_size(w):
    V1 = np.sum(w)
    V2 = np.sum(w**2)
    return V1**2 / V2
