import numpy as np
import statsmodels.api as sm
from scipy.odr import Model, Data, ODR
from scipy.optimize import minimize

import emcee

from ..utils import flat_and_remove_bad, all_asarray

def get_linear_model(k, b):
    def func(x):
        return k * x + b

    return func

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


def xy2logxy(x, y, x_err, y_err):
    logx = np.log10(x)
    logy = np.log10(y)
    logx_err = x_err / (x * np.log(10))
    logy_err = y_err / (y * np.log(10))
    return logx, logy, logx_err, logy_err


def get_string(k, b, order='xy'):
    sign = '-' if np.sign(b) == -1 else '+'
    if order == 'xy':
        return f'y = {k:.2f}x {sign} {np.abs(b):.2f} '
    elif order == 'yx':
        return f'x = {k:.2f}y {sign} {np.abs(b):.2f}'


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
    allasarray + flat_and_remove_bad
    '''
    return flat_and_remove_bad(all_asarray(xs))