import numpy as np
import statsmodels.api as sm
from scipy.odr import Model, Data, ODR
from scipy.optimize import minimize

import emcee

from ..utils import remove_bad, all_asarray, flag_bad


def get_linear_model(k, b):

    def func(x):
        return k * x + b

    return func


def get_linear_model_interval_bs(func_lst, alpha):

    def func_interval(x):
        y_lst = [func(x) for func in func_lst]
        y_l, y_u = np.percentile(y_lst,
                                 [alpha / 2 * 100, (1 - alpha / 2) * 100],
                                 axis=0)
        return y_l, y_u

    return func_interval


def get_OLS_bs(x, y, bs_N=0.8, bs_times=1000, return_res=False, alpha=0.32):
    if bs_N < 1:
        bs_N = x.shape[0] * bs_N

    results_lst = []

    for _ in range(bs_times):
        idx = np.random.choice(x.shape[0], int(bs_N), replace=False)
        res = get_OLS(x[idx], y[idx], return_res=return_res)
        results_lst.append(res)

    if return_res:
        return results_lst

    k_lst = [res['k'][0] for res in results_lst]
    b_lst = [res['b'][0] for res in results_lst]
    std_lst = [res['std'] for res in results_lst]
    func_lst = [res['func'] for res in results_lst]

    k = np.median(k_lst)
    b = np.median(b_lst)
    std = np.median(std_lst)
    k_l, k_u = np.percentile(k_lst, [alpha / 2 * 100, (1 - alpha / 2) * 100])
    b_l, b_u = np.percentile(b_lst, [alpha / 2 * 100, (1 - alpha / 2) * 100])
    std_l, std_u = np.percentile(std_lst,
                                 [alpha / 2 * 100, (1 - alpha / 2) * 100])

    func = get_linear_model(k, b)

    func_interval = get_linear_model_interval_bs(func_lst, alpha)

    return {
        'k': (k, k_l, k_u),
        'b': (b, b_l, b_u),
        'std': (std, std_l, std_u),
        'func': func,
        'func_interval': func_interval
    }


def get_OLS(x, y, return_res=False):
    # y = kx + b
    x, y = preprocess([x, y])

    X = sm.add_constant(x)
    model = sm.OLS(y, X)
    results = model.fit()

    if return_res:
        return results

    b, k = results.params
    b_err, k_err = results.HC1_se

    func = get_linear_model(k, b)

    std = np.std(func(x) - y)

    return {'k': (k, k_err), 'b': (b, b_err), 'std': std, 'func': func}


def get_OLS_nd(X, y):
    X, y = preprocess([X, y])
    X = sm.add_constant(X)
    model = sm.OLS(y, X)
    results = model.fit()

    def func(X):
        return results.predict(sm.add_constant(X))

    return results, func


def get_WLS(x, y, y_err, return_res=False):
    # y = kx + b
    x, y, y_err = preprocess([x, y, y_err])

    X = sm.add_constant(x)
    model = sm.WLS(y, X, weights=1. / np.square(y_err))
    results = model.fit()

    if return_res:
        return results

    b, k = results.params
    b_err, k_err = results.HC1_se

    func = get_linear_model(k, b)

    std = np.std(func(x) - y)

    return {'k': (k, k_err), 'b': (b, b_err), 'std': std, 'func': func}


def get_ODR(x, y, x_err=None, y_err=None, return_res=False):

    # https://docs.scipy.org/doc/scipy/reference/odr.html

    def f(B, x):
        return B[0] * x + B[1]

    if x_err is None:
        x_err = np.ones_like(x)
    if y_err is None:
        y_err = np.ones_like(y)

    x, y, x_err, y_err = preprocess([x, y, x_err, y_err])

    linear = Model(f)
    mydata = Data(x, y, wd=1. / np.square(x_err), we=1. / np.square(y_err))
    myodr = ODR(mydata, linear, beta0=[1., 0.])
    results = myodr.run()

    if return_res:
        return results

    k, b = results.beta
    k_err, b_err = results.sd_beta

    func_xy = get_linear_model(k, b)
    func_yx = get_linear_model(1 / k, -b / k)

    x_std = np.std(func_yx(y) - x)
    y_std = np.std(func_xy(x) - y)

    # 垂直距离的标准差
    d_std = np.std((func_xy(x) - y) / np.sqrt(k**2 + 1))

    return {
        'k': (k, k_err),
        'b': (b, b_err),
        'x_std': x_std,
        'y_std': y_std,
        'd_std': d_std,
        'func_xy': func_xy,
        'func_yx': func_yx
    }


def get_linear_regression(x, y, x_err, y_err):
    return {
        'OLS_xy': get_OLS(x, y),
        'OLS_yx': get_OLS(y, x),
        'WLS_xy': get_WLS(x, y, y_err),
        'WLS_yx': get_WLS(y, x, x_err),
        'ODR_nw': get_ODR(x, y),
        'ODR_w': get_ODR(x, y, x_err, y_err),
    }


def xy2logxy(x, y, x_err, y_err, no_invalid_value_warning=False):
    """
    Convert x, y, x_err, y_err to logx, logy, logx_err, logy_err; log means log10.

    Parameters
    ----------
    x, y, x_err, y_err : array_like

    no_invalid_value_warning : bool, optional
        If True, suppress RuntimeWarning: invalid value encountered in log10.
    
    Returns
    -------
    logx, logy, logx_err, logy_err : array_like
    """

    # Save the current error handling settings
    old_settings = np.seterr(
        invalid='ignore') if no_invalid_value_warning else None

    try:
        logx = np.log10(x)
        logy = np.log10(y)
        logx_err = x_err / (x * np.log(10))
        logy_err = y_err / (y * np.log(10))
    finally:
        # Restore the old error handling settings
        if no_invalid_value_warning:
            np.seterr(**old_settings)  # pylint: disable=not-a-mapping

    return logx, logy, logx_err, logy_err


def get_string(k,
               b,
               order='xy',
               x_name='x',
               y_name='y',
               k_err=None,
               b_err=None,
               format='.2f'):
    sign = '-' if np.sign(b) == -1 else '+'

    if order == 'xy':
        return f'{y_name} = {get_with_err_str(k, k_err, format=format)}{x_name} {sign} {get_with_err_str(np.abs(b), b_err, format=format)}'
    elif order == 'yx':
        return f'{x_name} = {get_with_err_str(k, k_err, format=format)}{y_name} {sign} {get_with_err_str(np.abs(b), b_err, format=format)}'


def get_with_err_str(value, err, format='.2f', err_style='pm_latex'):
    err_str = get_err_str(err, style=err_style, format=format)
    return f'{value:{format}}{err_str}'


def get_err_str(err, style='pm_latex', format='.2f'):

    if style not in ['pm_latex', 'pm_unicode', 'bracket']:
        raise ValueError(
            "style must be one of 'pm_latex', 'pm_unicode', 'bracket'")

    err_str = ''

    if err is not None:
        if style == 'pm_latex':
            err_str = f'$\pm${err:{format}}'
        elif style == 'pm_unicode':
            err_str = f'+/-{err:{format}}'
        elif style == 'bracket':
            err_str = f'({err:{format}})'

    return err_str


def get_ans_posterior(X, y, y_err=None):

    if y_err is not None:
        X = X / y_err[:, None]
        y = y / y_err

    beta = np.linalg.pinv(X.T @ X) @ X.T @ y
    Sigma = np.linalg.pinv(X.T @ X)

    return beta, Sigma


def maximum_likelihood(x, y, x_err, y_err, k0=1, b0=0, sig_int0=0):

    def log_likelihood(theta, x, y, x_err, y_err):
        '''
        y = kx + b
        sig = sqrt((k x_err)^2 + y_err^2 + sig_int^2)
        '''
        k, b, sig_int = theta
        model = k * x + b
        sigma2 = np.square(k * x_err) + np.square(y_err) + np.square(sig_int)
        return -0.5 * np.sum((y - model)**2 / sigma2 + np.log(sigma2))

    x, y, x_err, y_err = preprocess([x, y, x_err, y_err])

    initial = np.array([k0, b0, sig_int0])
    nll = lambda *args: -log_likelihood(*args)
    soln = minimize(nll, initial, args=(x, y, x_err, y_err))
    return soln


def mcmc_posterior(x,
                   y,
                   x_err,
                   y_err,
                   k_range=(-1e10, 1e10),
                   b_range=(-1e10, 1e10),
                   sig_int_range=(0, 1e10),
                   emcee_kwargs=None):

    def log_likelihood(theta, x, y, x_err, y_err):
        '''
        y = kx + b
        sig = sqrt((k x_err)^2 + y_err^2 + sig_int^2)
        '''
        k, b, sig_int = theta
        model = k * x + b
        sigma2 = np.square(k * x_err) + np.square(y_err) + np.square(sig_int)
        return -0.5 * np.sum((y - model)**2 / sigma2 + np.log(sigma2))

    def log_prior(theta):
        k, b, sig_int = theta
        return 0.0 if k_range[0] <= k <= k_range[1] and b_range[
            0] <= b <= b_range[1] and sig_int_range[
                0] <= sig_int <= sig_int_range[1] else -np.inf

    def log_probability(theta, x, y, x_err, y_err):
        lp = log_prior(theta)
        return lp + log_likelihood(theta, x, y, x_err,
                                   y_err) if np.isfinite(lp) else -np.inf

    x, y, x_err, y_err = preprocess([x, y, x_err, y_err])

    if emcee_kwargs is None:
        emcee_kwargs = {}

    nwalkers = emcee_kwargs.get('nwalkers', 8)
    ndim = 3

    sampler = emcee.EnsembleSampler(nwalkers,
                                    ndim,
                                    log_probability,
                                    args=(x, y, x_err, y_err))

    pos = np.array([
        np.random.uniform(k_range[0], k_range[1], size=(nwalkers, )),
        np.random.uniform(b_range[0], b_range[1], size=(nwalkers, )),
        np.random.uniform(sig_int_range[0],
                          sig_int_range[1],
                          size=(nwalkers, ))
    ])

    sampler.run_mcmc(pos.T,
                     emcee_kwargs.get('steps', 5000),
                     progress=emcee_kwargs.get('progress', True))

    return sampler


def preprocess(xs):
    '''
    allasarray + remove_bad
    '''
    return remove_bad(all_asarray(xs))
