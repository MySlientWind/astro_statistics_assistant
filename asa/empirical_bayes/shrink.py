import numpy as np


def shrink_gg(x_obs, x_err):
    # Tweedie's formula when p(x_obs) is considered to be a normal distribution
    mu = np.mean(x_obs)
    sigma2 = np.var(x_obs)
    B = (sigma2 - np.square(x_err))/sigma2
    return B * x_obs + (1 - B) * mu




