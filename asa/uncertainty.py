'''
https://en.wikipedia.org/wiki/Propagation_of_uncertainty
'''

import numpy as np


def sum(xerrs, w=None):
    '''return the error of sum of x'''
    xerrs = np.asarray(xerrs)
    if w is not None:
        xerrs = xerrs * w/np.sum(w)
    return np.sqrt(np.sum(np.square(xerrs), axis=0))


def ratio(x, y, xerr, yerr):
    return np.sqrt(np.square(xerr / y) + np.square((x * yerr) / np.square(y)))


def multiply(x, y, xerr, yerr):
    '''calculate the error of xy'''
    return np.sqrt(np.square(xerr * y) + np.square(x * yerr))


def log(x, xerr):
    '''caluculate the error of log(x)'''
    return np.abs(xerr / x)


def log10(x, x_err):
    '''caluculate the error of log10(x)'''
    return np.abs(x_err / x / np.log10(x))

def exp(x, x_err):
    '''caluculate the error of exp(x)'''
    return np.abs(x_err * np.exp(x))






