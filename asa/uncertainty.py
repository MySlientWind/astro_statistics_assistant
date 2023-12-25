'''
https://en.wikipedia.org/wiki/Propagation_of_uncertainty
'''

# TODO: test cover all
# TODO: sqrt

import numpy as np

# 1 element

def log(x, x_err, with_value=False):
    # sourcery skip: assign-if-exp, reintroduce-else
    '''
    caluculate the error of log(x)
    |x_err/x|
    '''
    if with_value:
        return np.abs(x_err / x), np.log(x)
    return np.abs(x_err / x)


def log10(x, x_err, with_value=False):
    # sourcery skip: assign-if-exp, reintroduce-else
    '''
    caluculate the error of log10(x)
    |x_err/(log(10) * x)|
    '''
    if with_value:
        return np.abs(x_err / x / np.log(10)), np.log10(x)
    return np.abs(x_err / x / np.log(10))


def log10_snr(x, x_snr):
    '''
    caluculate the snr of log10(x)
    
    |log10(x) * (x_snr * log(10)|
    '''
    return np.log10(x) * (x_snr * np.log(10))


def exp(x, x_err, with_value=False):
    '''
    caluculate the error of exp(x)
    
    |x_err * exp(x)|
    '''
    # sourcery skip: assign-if-exp, reintroduce-else
    value = np.exp(x)
    if with_value:
        return np.abs(x_err * value), value
    return np.abs(x_err * value)

def power(x, x_err, a=1, with_value=False):
    '''
    caluculate the error of x**a
    
    |a * x**(a-1) * x_err|
    '''
    if a < 0:
        # convert x to float
        x = np.asarray(x)
        x = x.astype('float')

    if with_value:
        return np.abs(a * np.power(x, a - 1) * x_err), np.power(x, a)
    return np.abs(a * np.power(x, a - 1) * x_err)

def power_snr(x, x_snr, a=1):
    '''
    caluculate the snr of x**a
    
    |x_snr/a|
    '''
    return np.abs(x_snr/a)

def square(x, x_err, with_value=False):
    return power(x, x_err, a=2, with_value=with_value)

def square_snr(x, x_snr):
    return power_snr(x, x_snr, a=2)

# 2 elements

def ratio(x, y, x_err, y_err, with_value=False):
    # sourcery skip: assign-if-exp, reintroduce-else
    '''
    The error of x/y
    |x/y| * ((x_err/x)**2 + (y_err/y)**2)**0.5
    or
    ((x_err/y)**2 + (y_err * x/y**2)**2)**0.5
    '''
    err = np.sqrt(np.square(x_err / y) + np.square((x * y_err) / np.square(y)))
    if with_value:
        return err, x / y
    return err


def multiply(x, y, x_err, y_err, with_value=False):
    # sourcery skip: assign-if-exp, reintroduce-else
    '''
    calculate the error of xy
    sqrt(x**2 * y_err**2 + y**2 * x_err**2)
    '''
    err = np.sqrt(np.square(x_err * y) + np.square(x * y_err))
    if with_value:
        return err, x * y
    return err

# n elements

def sum(xs, x_errs, w=None, a=None, with_value=False):
    # sourcery skip: assign-if-exp, reintroduce-else
    '''
    return the error of
    sum(x_i * w_i/sum(w_i) * a_i)
    = sum((x_err_i * w_i/sum(w_i) * a_i)**2)**0.5
    '''
    x_errs = np.asarray(x_errs)
    if a is None:
        a = np.ones_like(x_errs)
    if w is not None:
        a = a * w / np.sum(w)
    x_errs = x_errs * a
    err = np.sqrt(np.sum(np.square(x_errs), axis=0))
    if with_value:
        return err, np.sum(xs * a, axis=0)
    return err