from functools import partial
import warnings

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.weightstats import DescrStatsW
from scipy.stats import binned_statistic

from .Bcorner import corner, hist2d, quantile
from .utils import flag_bad, auto_set_range, is_empty
from .binning_methods import weighted_binned_statistic, bin_2d
from .loess2d import loess_2d_map

# TODO: extract common code
# - flag and remove bad


def plot_trend(x,
               y,
               bins=20,
               ytype='median',
               fig=None,
               ax=None,
               range=None,
               auto_p=None,
               weights=None,
               N_min=1,
               lowlim=25,
               uplim=75,
               fkind=None,
               prop_kwargs=None,
               errorbar_kwargs=None,
               fbetween_kwargs=None,
               plot_kwargs=None):  # sourcery skip: avoid-builtin-shadow
    """
    Make a plot to show the trend between x and y

    Parameters
    -----------------
    x : array_like[nsamples,]                                     
        The samples.                             
                                                           
    y : array_like[nsamples,]                            
        The samples.
                                                                     
    ytype: Character string or float
        The y value used to plot
        The available character string is "median" or 'mean'. If ytype is set as "median", the trend is shown by the median value of y as a function of x.
        if ytype is float, y_value = np.percentile(y, ytype)
        default: "median"


    range: array_like[2, 2] or string ([x_min, x_max], [y_min, y_max]), 
        if not 'auto', the range is automatically determined according to quantile specified by auto_p, if 'auto'                                                        
        default: 'auto'

    auto_p: array_like[2, 2] or string
       Used to generate range if range == 'auto'
       x_min = np.percentile(x, auto_p[0][0])
       x_max = np.percentile(x, auto_p[0][1])
       y_min = np.percentile(y, auto_p[1][0])
       y_max = np.percentile(y, auto_p[1][1])
       default: ([1, 99], [1, 99])

    weights: Optional[array_like[nsamples,]] 
        An optional weight corresponding to each sample.

    N_min: int
        The minimum number of samples in each bin to plot the trend
    
    ax : matplotlib.Axes
        A axes instance on which to add the line.

    ifscatter: whether to plot scatter

    uplim (%): The upper limit of the scatter, in [0, 100]

    lowlim (%): The lower limit of the scatter, in [0, 100]

    fkind: which ways to show the scatter, "errorbar" and "fbetween" are available
        
    plot_kwargs: function in ``matplotlib``


    prop_kwargs: dict (to be added)
        The extra property used to constrain the x, y, data
        props : array_like[nsamples,]                                     
            The samples with size same as x/y.                             
        pmax : the maximum value of props
        pmin : the minimum value of props
        
    plot_scatter_kwargs: dict
        to describe the scatter
        function in ``matplotlib``
    
    """

    if ax is None:
        ax = plt.gca()

    if plot_kwargs is None:
        plot_kwargs = {}

    if prop_kwargs is not None:
        props = prop_kwargs["props"]
        pmin = prop_kwargs.get("pmin", min(props))
        pmax = prop_kwargs.get("pmax", max(props))
        prop_index = (props >= pmin) & (props <= pmax)
        x = x[prop_index]
        y = y[prop_index]
        # print(np.shape(x), np.shape(y))

    ifscatter = fkind is not None

    bad = flag_bad(x) | flag_bad(y)
    x = x[~bad]
    y = y[~bad]

    if is_empty(x) or is_empty(y):
        warnings.warn("The x or y are empty after remove bad data skip the plot")
        return 

    range = auto_set_range(x, y, range, auto_p)

    xrange = range[0]
    yrange = range[1]

    if weights is None:
        weights = np.ones_like(x)

    func_median = lambda y, w: quantile(y, 0.5, weights=w)
    func_mean = lambda y, w: np.average(y, weights=w).reshape(1)

    # TODO: extract as a standalone method, support more statistic

    is_y_in_range = (y > yrange[0]) & (y < yrange[1])

    loads = [
        weighted_binned_statistic(x[is_y_in_range],
                                  x[is_y_in_range],
                                  weights[is_y_in_range],
                                  statistic=func_median,
                                  bins=bins,
                                  range=xrange)
    ]

    if ytype == "median":
        y_statistic = func_median
    elif ytype == "mean":
        y_statistic = func_mean

    statistic_list = [y_statistic]

    if ifscatter:
        lower_statistic = lambda y, w: quantile(y, q=lowlim / 100, weights=w)
        upper_statistic = lambda y, w: quantile(y, q=uplim / 100, weights=w)

        statistic_list.append(lower_statistic)
        statistic_list.append(upper_statistic)

    for statistic in statistic_list:
        _value = weighted_binned_statistic(x[is_y_in_range],
                                           y[is_y_in_range],
                                           weights[is_y_in_range],
                                           statistic=statistic,
                                           bins=bins,
                                           range=xrange)
        loads.append(_value)

    loads = np.hstack(loads)

    N_in_each_bin, _, _ = binned_statistic(x[is_y_in_range],
                                           x[is_y_in_range],
                                           statistic='count',
                                           bins=bins,
                                           range=xrange)
    loads[N_in_each_bin < N_min] = np.nan

    ax.plot(loads[:, 0], loads[:, 1], **plot_kwargs)

    if ifscatter:
        if fkind == "errorbar":
            if errorbar_kwargs is None:
                errorbar_kwargs = {}
            ax.errorbar(loads[:, 0],
                        loads[:, 1],
                        yerr=(loads[:, 3] - loads[:, 2]) / 2.0,
                        **errorbar_kwargs)
        elif fkind == "fbetween":
            if fbetween_kwargs is None:
                fbetween_kwargs = {}
                fbetween_kwargs["color"] = plot_kwargs.get("color", "r")
                fbetween_kwargs["alpha"] = 0.2
            else:
                if "color" not in fbetween_kwargs:
                    fbetween_kwargs["color"] = plot_kwargs.get("color", "r")
                if "alpha" not in fbetween_kwargs:
                    fbetween_kwargs["alpha"] = 0.2
            ax.fill_between(loads[:, 0], loads[:, 3], loads[:, 2],
                            **fbetween_kwargs)


def plot_scatter(x,
                 y,
                 z=None,
                 fig=None,
                 ax=None,
                 range=None,
                 auto_p=None,
                 weights=None,
                 label=None,
                 ifsmooth=False,
                 smooth_kwargs=None,
                 plot_kwargs=None):  # sourcery skip: avoid-builtin-shadow

    # TODO: z_range, automatically adjust?
    # usage of xnew / ynew

    hasz = True
    if z is None:
        z = np.ones(len(x))
        hasz = False

    if weights is None:
        weights = np.ones_like(x)

    bad = flag_bad(x) | flag_bad(y) | flag_bad(z)
    x = x[~bad]
    y = y[~bad]
    z = z[~bad]

    if is_empty(x) or is_empty(y) or is_empty(z):
        warnings.warn("The x, y or z are empty after remove bad data skip the plot")
        return 

    range = auto_set_range(x, y, range, auto_p)

    xrange = range[0]
    yrange = range[1]

    is_in_range = (x >= xrange[0]) & (x <= xrange[1]) & (y >= yrange[0]) & (
        y <= yrange[1])

    if plot_kwargs is None:
        plot_kwargs = {}

    if ifsmooth:
        if smooth_kwargs is None:
            smooth_kwargs = {}
        nsmooth = smooth_kwargs.get("nsmooth", 0.5)
        xnew = x[is_in_range].copy()
        ynew = y[is_in_range].copy()
        znew = loess_2d_map(x[is_in_range], y[is_in_range], z[is_in_range],
                            xnew, ynew, weights[is_in_range], nsmooth)
        sc = ax.scatter(xnew, ynew, c=znew, label=label, **plot_kwargs)
        plt.colorbar(sc, ax=ax)

    else:
        if hasz:
            sc = ax.scatter(x[is_in_range],
                            y[is_in_range],
                            c=z[is_in_range],
                            label=label,
                            **plot_kwargs)
            plt.colorbar(sc, ax=ax)
        else:
            ax.scatter(x[is_in_range],
                       y[is_in_range],
                       label=label,
                       **plot_kwargs)


def plot_corner(xs,
                bins=20,
                range=None,
                weights=None,
                color="k",
                hist_bin_factor=1,
                kde_smooth=False,
                kde_smooth1d=False,
                smooth=None,
                smooth1d=None,
                labels=None,
                label_kwargs=None,
                show_titles=False,
                title_fmt=".2f",
                title_kwargs=None,
                truths=None,
                truth_color="#4682b4",
                scale_hist=False,
                quantiles=None,
                verbose=False,
                fig=None,
                max_n_ticks=5,
                top_ticks=False,
                use_math_text=False,
                reverse=False,
                hist_kwargs=None,
                plot_add=None,
                plot_add_1d=None,
                dpi=None,
                **hist2d_kwargs):
    corner(xs,
           bins=bins,
           range=range,
           weights=weights,
           color=color,
           hist_bin_factor=hist_bin_factor,
           kde_smooth=kde_smooth,
           kde_smooth1d=kde_smooth1d,
           smooth=smooth,
           smooth1d=smooth1d,
           labels=labels,
           label_kwargs=label_kwargs,
           show_titles=show_titles,
           title_fmt=title_fmt,
           title_kwargs=title_kwargs,
           truths=truths,
           truth_color=truth_color,
           scale_hist=scale_hist,
           quantiles=quantiles,
           verbose=verbose,
           fig=fig,
           max_n_ticks=max_n_ticks,
           top_ticks=top_ticks,
           use_math_text=use_math_text,
           reverse=reverse,
           hist_kwargs=hist_kwargs,
           plot_add=plot_add,
           plot_add_1d=plot_add_1d,
           dpi=dpi,
           **hist2d_kwargs)


def plot_contour(x,
                 y,
                 bins=20,
                 range='auto',
                 kde_smooth=False,
                 auto_p=None,
                 weights=None,
                 levels=None,
                 smooth=None,
                 ax=None,
                 color=None,
                 quiet=False,
                 plot_datapoints=True,
                 plot_density=True,
                 plot_contours=True,
                 no_fill_contours=False,
                 fill_contours=False,
                 contour_kwargs=None,
                 contourf_kwargs=None,
                 data_kwargs=None,
                 pcolor_kwargs=None):
    hist2d(x,
           y,
           bins=bins,
           range=range,
           kde_smooth=kde_smooth,
           auto_p=auto_p,
           weights=weights,
           levels=levels,
           smooth=smooth,
           ax=ax,
           color=color,
           quiet=quiet,
           plot_datapoints=plot_datapoints,
           plot_density=plot_density,
           plot_contours=plot_contours,
           no_fill_contours=no_fill_contours,
           fill_contours=fill_contours,
           contour_kwargs=contour_kwargs,
           contourf_kwargs=contourf_kwargs,
           data_kwargs=data_kwargs,
           pcolor_kwargs=pcolor_kwargs)


def plot_heatmap(x,
                 y,
                 z,
                 weights=None,
                 ax=None,
                 bins=10,
                 min_data=0,
                 range=None,
                 auto_p=None,
                 map_kind='pcolor',
                 pcolor_kwargs=None,
                 contour_kwargs=None,
                 set_clabel=False,
                 clabel_kwargs=None):  # sourcery skip: avoid-builtin-shadow
    """
    kind: 'pcolor' or 'contour'
    """
    # TODO: z_range
    # TODO: weights

    bad = flag_bad(x) | flag_bad(y) | flag_bad(z)
    x = x[~bad]
    y = y[~bad]
    z = z[~bad]

    if is_empty(x) or is_empty(y) or is_empty(z):
        warnings.warn("The x, y or z are empty after remove bad data skip the plot")
        return 

    # TODO: z range
    range = auto_set_range(x, y, range, auto_p)

    X, Y, Z, x_edges, y_edges = bin_2d(x,
                                       y,
                                       z,
                                       bins,
                                       min_data=min_data,
                                       range=range)
    if ax is None:
        ax = plt.gca()

    if map_kind == 'pcolor':
        if pcolor_kwargs is None:
            pcolor_kwargs = {}

        # Maybe use pcolormesh for high performance?
        # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.pcolormesh.html#differences-pcolor-pcolormesh
        ax.pcolor(x_edges, y_edges, Z, **pcolor_kwargs)

    elif map_kind == 'contour':

        if contour_kwargs is None:
            contour_kwargs = {}
        cont = ax.contour(X, Y, Z, **contour_kwargs)
        if set_clabel:
            if clabel_kwargs is None:
                clabel_kwargs = {}
            ax.clabel(cont, **clabel_kwargs)

        return cont


def plot_sample_to_point(x,
                         y,
                         ax=None,
                         weights=None,
                         ddof=0,
                         center_type='mean',
                         error_type='std',
                         quantiles=None,
                         errorbar_kwargs=None):
    """
    The function to summarize a sample to a point and then plot it
    

    center_type: 'mean' or 'median'
    error_type: 'std', 'std_mean' or 'quantile'
    
    """
    bad = flag_bad(x) | flag_bad(y)
    x = x[~bad]
    y = y[~bad]

    if ax is None:
        ax = plt.gca()

    if errorbar_kwargs is None:
        errorbar_kwargs = {}

    if center_type == 'mean':
        weighted_stats = DescrStatsW(np.c_[x, y], weights=weights, ddof=ddof)
        x_cen = weighted_stats.mean[0]
        y_cen = weighted_stats.mean[1]
    elif center_type == 'median':
        x_cen = quantile(x, q=0.5, weights=weights)[0]
        y_cen = quantile(y, q=0.5, weights=weights)[0]
    else:
        raise ValueError('center_type must be one of mean or median')

    if error_type == 'std':
        weighted_stats = DescrStatsW(np.c_[x, y], weights=weights, ddof=ddof)
        x_err = weighted_stats.std[0]
        y_err = weighted_stats.std[1]
    elif error_type == 'std_mean':
        weighted_stats = DescrStatsW(np.c_[x, y], weights=weights, ddof=ddof)
        x_err = weighted_stats.std_mean[0]
        y_err = weighted_stats.std_mean[1]
    elif error_type == 'quantile':
        if quantiles is None:
            quantiles = [0.16, 0.84]
        x_quant = quantile(x, q=quantiles, weights=weights).reshape(-1, 1)
        y_quant = quantile(y, q=quantiles, weights=weights).reshape(-1, 1)
        x_err = np.abs(x_quant - x_cen)
        y_err = np.abs(y_quant - y_cen)
    else:
        raise ValueError('error_type must be one of std, std_mean or quantile')

    ax.errorbar(x_cen, y_cen, xerr=x_err, yerr=y_err, **errorbar_kwargs)


def plot_line(x=None,
              y=None,
              p1=None,
              p2=None,
              k=None,
              b=None,
              ax=None,
              **kwargs):

    if ax is None:
        ax = plt.gca()

    if x is not None:
        ax.axvline(x, **kwargs)
        return

    if y is not None:
        ax.axhline(y, **kwargs)
        return

    if p1 is not None:
        if p2 is not None:
            ax.axline(p1, p2, **kwargs)
            return
        elif k is not None:
            ax.axline(p1, slope=k, **kwargs)
            return
        elif b is not None:
            ax.axline(p1, slope=(p1[1] - b) / p1[0], **kwargs)
            return

    # sourcery skip: merge-nested-ifs
    if k is not None:
        if b is not None:
            ax.axline((0, b), slope=k, **kwargs)
            return

    raise ValueError("Invalid input")
