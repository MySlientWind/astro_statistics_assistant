'''
https://en.wikipedia.org/wiki/Propagation_of_uncertainty
'''

# TODO: test cover all

import numpy as np

# 1 element

def log(x, x_err):
    '''
    caluculate the error of log(x)
    |x_err/x|
    '''
    return np.abs(x_err / x)


def log10(x, x_err):
    '''
    caluculate the error of log10(x)
    |x_err/(log(10) * x)|
    '''
    return np.abs(x_err / x / np.log(10))


def log10_snr(x, x_snr):
    '''
    caluculate the snr of log10(x)
    
    |log10(x) * (x_snr * log(10)|
    '''
    return np.log10(x) * (x_snr * np.log(10))


def exp(x, x_err):
    '''
    caluculate the error of exp(x)
    
    |x_err * exp(x)|
    '''
    return np.abs(x_err * np.exp(x))

def power(x, x_err, a=1):
    '''
    caluculate the error of x**a
    
    |a * x**(a-1) * x_err|
    '''
    if a < 0:
        # convert x to float
        x = np.asarray(x)
        x = x.astype('float')

    return np.abs(a * np.power(x, a - 1) * x_err)

def power_snr(x, x_snr, a=1):
    '''
    caluculate the snr of x**a
    
    |x_snr/a|
    '''
    return np.abs(x_snr/a)

def square(x, x_err):
    return power(x, x_err, a=2)

def square_snr(x, x_snr):
    return power_snr(x, x_snr, a=2)

# 2 elements

def ratio(x, y, x_err, y_err):
    '''
    The error of x/y
    |x/y| * ((x_err/x)**2 + (y_err/y)**2)**0.5
    or
    ((x_err/y)**2 + (y_err * x/y**2)**2)**0.5
    '''
    return np.sqrt(np.square(x_err / y) + np.square((x * y_err) / np.square(y)))


def multiply(x, y, x_err, y_err):
    '''
    calculate the error of xy
    sqrt(x**2 * y_err**2 + y**2 * x_err**2)
    '''
    return np.sqrt(np.square(x_err * y) + np.square(x * y_err))

# n elements

def sum(x_errs, w=None, a=None):
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
    return np.sqrt(np.sum(np.square(x_errs), axis=0))