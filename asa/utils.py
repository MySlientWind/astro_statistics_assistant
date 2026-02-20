from __future__ import annotations

import sys
import inspect
from typing import Any, Dict, List, Mapping

import numpy as np

_range = range


def flag_bad(x):
    """
    It returns True if the input is NaN or Inf, and False otherwise

    :param x: the input data
    :return: A boolean array of the same shape as x, where True indicates
    that the corresponding element of x is NaN or +/-inf.
    """
    return np.isnan(x) | np.isinf(x)

# TODO: move to sklearn_tools
def balance_class(x, y, random_state=None):
    """
    Balance the classes in a dataset by randomly removing data points from the majority class(es).

    :param x: the input data
    :param y: the labels
    :param random_state: the random state to use for the random number generator
    :return: the balanced data and labels
    """
    if random_state is not None:
        np.random.seed(random_state)
    unique, counts = np.unique(y, return_counts=True)
    min_count = np.min(counts)
    idx = np.concatenate([
        np.random.choice(np.where(y == u)[0], min_count, replace=False)
        for u in unique
    ])
    return x[idx], y[idx]


def _validate_type(_type):
    if _type not in ['both', 'number', 'array']:
        raise ValueError(
            f"type should be 'both', 'number' or 'array', got {_type}")


def is_float(x, _type='both'):
    '''
    check if x is kind of float, such as built-in float, 
    np.float32, np.float64, np.ndarray of float, list of float...
    '''

    _validate_type(_type)

    flag = False
    if _type in ['both', 'number']:
        flag |= isinstance(x, (float, np.float32, np.float64))
    if _type in ['both', 'array']:
        flag |= ((isinstance(x, np.ndarray) and x.dtype.kind == 'f') or
                 (isinstance(x, list) and all(isinstance(y, float)
                                              for y in x)))
    return flag


def is_int(x, _type='both'):
    '''
    check if x is kind of int, such as built-in int, 
    np.int32, np.int64, np.ndarray of int, list of int...
    '''
    _validate_type(_type)

    flag = False
    if _type in ['both', 'number']:
        flag |= isinstance(x, (int, np.int32, np.int64))
    if _type in ['both', 'array']:
        flag |= ((isinstance(x, np.ndarray) and x.dtype.kind == 'i') or
                 (isinstance(x, list) and all(isinstance(y, int) for y in x)))

    return flag

#TODO: support _type
def is_bool(x):
    '''
    check if x is kind of bool, such as built-in bool, np.bool, np.ndarray of bool, list of bool...
    '''
    return isinstance(
        x, bool) or (isinstance(x, np.ndarray) and x.dtype.kind == 'b') or (
            isinstance(x, list) and all(isinstance(y, bool) for y in x))

def is_real_number(x, _type='both'):
    '''
    check if x is kind of float or int, such as built-in float, int, 
    np.float32, np.float64, np.int32, np.int64, np.ndarray of float or int, list of float or int...
    '''
    return is_float(x, _type=_type) or is_int(x, _type=_type)


def string_to_list(string):
    return [string] if isinstance(string, str) else string


def is_string_or_list_of_string(x):
    return (isinstance(x, str)
            or isinstance(x, list) and all(isinstance(y, str) for y in x))


def list_reshape(lst: List, shape) -> List[List]:
    """
    Reshape a list into a list of lists
    :param lst: the input list
    :param shape: the shape of the output list
    :return: a list of lists
    """
    return [lst[i:i + shape[1]] for i in _range(0, len(lst), shape[1])]


def set_range_default(x):
    if len(x) == 0:
        return [0, 1]
    elif len(x) == 1:
        return [x[0] - 0.5, x[0] + 0.5]
    else:
        return [np.min(x), np.max(x)]


def auto_set_range(*args, _range=None, auto_p=None):

    DEFAULT_AUTO_P = (1, 99)

    def is_1d_range(lst):
        flag = (len(lst) == 2)
        if flag:
            flag &= is_real_number(lst[0], _type='number')
        if flag:
            flag &= is_real_number(lst[1], _type='number')
        if flag:
            flag &= (lst[0] <= lst[1])
            return flag
        return False

    n_arg = len(args)

    # case: None
    if _range is None:
        _range = [set_range_default(x) for x in args]
    # otherwise, auto_p will be used
    else:
        if auto_p is None:
            auto_p = (DEFAULT_AUTO_P for _ in range(n_arg))
        # case: [min, max]
        elif is_1d_range(auto_p):
            auto_p = [auto_p for _ in range(n_arg)]
        # case: [[min, max]|None, ...]
        elif len(auto_p) == n_arg:
            for i, p in enumerate(auto_p):
                if p is None:
                    auto_p[i] = DEFAULT_AUTO_P
        else:
            raise ValueError(f"auto_p ({auto_p}) can not be parsed")

    # Now auto_p is [[min, max], ...]
    # case: 'auto'
    if _range == 'auto':
        _range = []
        for x, p in zip(args, auto_p):
            _range.append([np.percentile(x, p[0]), np.percentile(x, p[1])])
    # case: [min, max]
    elif is_1d_range(_range):
        _range = [_range for _ in range(n_arg)]
    # case: [[min, max]|None|'auto', ...]
    elif len(_range) == n_arg:
        for i, r in enumerate(_range):
            if r is None:
                _range[i] = set_range_default(args[i])
            elif r == 'auto':
                p = auto_p[i]
                x = args[i]
                _range[i] = [np.percentile(x, p[0]), np.percentile(x, p[1])]
            elif is_1d_range(r):
                pass
            else:
                raise ValueError(f"_range element {i} ({r}) can not be parsed")
    else:
        raise ValueError(f"_range ({_range}) can not be parsed")

    if (len(_range) == 1) and (n_arg == 1):
        _range = _range[0]

    return _range


def get_kwargs_each(fixed_kwargs, changed_kwargs, shape):
    """
    Get a list of kwargs for each element in a 2D array

    :param fixed_kwargs: a dictionary of fixed kwargs
    :param changed_kwargs: a dictionary of changed kwargs
    :param shape: the shape of the 2D array
    :return: a list of kwargs
    """
    kwargs_each = []
    for i in _range(shape[0]):
        kwargs_each.append([])
        for j in _range(shape[1]):
            kwargs_each[-1].append({})
            for key, value in fixed_kwargs.items():
                kwargs_each[-1][-1][key] = value
            for key, value in changed_kwargs.items():
                kwargs_each[-1][-1][key] = value[i][j]
    return kwargs_each


def remove_bad(xs: List[np.ndarray],
               report=False,
               to_transpose=None) -> List[np.ndarray | None]:

    if to_transpose is None:
        to_transpose = []

    if xs[0].ndim == 1:
        n_sample = len(xs[0])
    else:
        n_sample = xs[0].shape[0]
    bad = np.zeros(n_sample, dtype=bool)

    for i, x in enumerate(xs):
        if x is None:
            continue
        if i in to_transpose:
            x = x.T
        if x.ndim == 1:
            bad |= flag_bad(x)
        else:
            bad |= flag_bad(x).any(axis=1)
    if report and np.any(bad):
        print(f"Bad sample detected: {np.where(bad)[0]}")

    res_lst = []
    for i, x in enumerate(xs):
        if x is None:
            res_lst.append(None)
        elif i in to_transpose:
            res_lst.append(x.T[~bad].T)
        else:
            res_lst.append(x[~bad])

    return res_lst


def all_asarray(xs):
    return [np.asarray(x) if x is not None else None for x in xs]


def is_empty(x):
    return len(x) == 0


def any_empty(xs):
    return any(is_empty(x) if x is not None else False for x in xs)


def all_subsample(xs, idx):
    return [x[idx] if x is not None else None for x in xs]


def get_ndim(x):
    return np.asarray(x).ndim


def get_shape(x) -> tuple:
    """
    Get the shape of a list of lists
    
    Parameters
    ----------
    lst : array-like
        The input list of lists.
    
    Returns 
    -------
    tuple
        The shape of the list.
    """
    return np.asarray(x).shape


def get_rank(x):
    return np.argsort(np.argsort(x))


def to_little_endian(x):
    """
    It takes a numpy array and returns a new numpy array with little-endian format.
    It checks if the array is already in little-endian format and returns it directly if so.

    :param x: A numpy array.
    :return: A numpy array in little-endian byte order.
    """
    x = np.array(x)
    if (x.dtype.byteorder == '<') or (x.dtype.byteorder == '='
                                      and sys.byteorder == 'little'):
        # If the array is already little-endian, return it as is.
        return x
    else:
        # Otherwise, convert to little-endian.
        return x.byteswap().newbyteorder('little')


def deduplicate(x_o, max_dx=0.1):

    # raise if not sorted
    if not np.all(np.diff(x_o) >= 0):
        raise ValueError("Input array must be sorted")

    x = np.asarray(x_o)
    x_last = x[0]
    i_last = 0

    for i, xi in enumerate(x):
        if xi == x_last:
            continue
        elif i - i_last > 1:
            dx = min(max_dx, xi - x_last)
            delta = dx * np.arange(1, i - i_last) / (i - i_last)
            x[i_last + 1:i] += delta
        x_last = xi
        i_last = i

    if i_last < len(x) - 1:
        dx = max_dx
        delta = dx * np.arange(1, len(x) - i_last) / (len(x) - i_last)
        x[i_last + 1:] += delta

    return x


def set_default_kwargs(kwargs: Mapping[str, Any] | None,
                       **defaults) -> Dict[str, Any]:
    """
    Return a new dict with defaults merged in; leave `kwargs` untouched.
    Raises TypeError if kwargs is not a mapping.

    Parameters
    ----------
    kwargs : dict | None
        The keyword arguments to be set.
        Regarded as an empty dict if None.
    defaults : dict
        The default keyword arguments.

    Returns
    -------
    dict
        The updated keyword arguments.
    """
    if kwargs is None:
        kwargs = {}
    if not isinstance(kwargs, Mapping):
        raise TypeError(
            f'kwargs must be a dict-like mapping, got {type(kwargs)}')

    # shallow copy of kwargs
    new_kwargs = dict(kwargs)
    for k, v in defaults.items():
        new_kwargs.setdefault(k, v)

    return new_kwargs


def ensure_parameter_spec(func,
                          param_name,
                          expected_default=inspect.Parameter.empty):
    """
    Validate if the function signature matches the decorator requirements.
    
    Args:
        func: The function to inspect.
        param_name: Name of the parameter to look for.
        expected_default: The value we expect the parameter to have as default.
        strict_default: If True, the parameter MUST have the expected_default.
                        If False, we only check if the parameter exists.
    """
    sig = inspect.signature(func)
    params = sig.parameters

    # Check 1: Existence
    if param_name not in params:
        raise ValueError(
            f"Function '{func.__name__}' is missing the required parameter: '{param_name}'"
        )

    # Check 2: Default Value (only if strict_default is requested)
    if expected_default is not inspect.Parameter.empty:
        actual_default = params[param_name].default
        if actual_default is not expected_default:
            raise ValueError(
                f"Function '{func.__name__}' parameter '{param_name}' must have a "
                f"default value of {expected_default}, but found {actual_default}."
            )
