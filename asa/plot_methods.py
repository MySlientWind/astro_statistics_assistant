from functools import partial

import numpy as np
import matplotlib.pyplot as plt
# from scipy.stats import binned_statistic

from .Bcorner import corner, hist2d, quantile
from .utils import flag_bad, weighted_binned_statistic


def plot_trend(x,
               y,
               bins=20,
               ytype='median',
               fig=None,
               ax=None,
               range=None,
               auto_p=None,
               weights=None,
               ifscatter=False,
               lowlim=25,
               uplim=75,
               fkind="fbetween",
               prop_kwargs=None,
               plot_scatter_kwargs=None,
               plot_kwargs=None):
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

    if ifscatter:
        if plot_scatter_kwargs is None:
            plot_scatter_kwargs = {}
            plot_scatter_kwargs["color"] = plot_kwargs.get("color", "r")
            plot_scatter_kwargs["alpha"] = 0.2
        else:
            if "color" not in plot_scatter_kwargs:
                plot_scatter_kwargs["color"] = plot_kwargs.get("color", "r")
            if "alpha" not in plot_scatter_kwargs:
                plot_scatter_kwargs["alpha"] = 0.2

    if prop_kwargs is not None:
        props = prop_kwargs["props"]
        pmin = prop_kwargs.get("pmin", min(props))
        pmax = prop_kwargs.get("pmax", max(props))
        prop_index = (props >= pmin) & (props <= pmax)
        x = x[prop_index]
        y = y[prop_index]
        # print(np.shape(x), np.shape(y))

    bad = flag_bad(x) | flag_bad(y)
    x = x[~bad]
    y = y[~bad]

    if range is None:
        xrange = [x.min(), x.max()]
        yrange = [y.min(), y.max()]
    elif range == 'auto':
        if auto_p is None:
            auto_p = ([1, 99], [1, 99])
        xrange = [
            np.percentile(x, auto_p[0][0]),
            np.percentile(x, auto_p[0][1])
        ]
        yrange = [
            np.percentile(y, auto_p[1][0]),
            np.percentile(y, auto_p[1][1])
        ]
    else:
        xrange = range[:][0]
        yrange = range[:][1]

    if weights is None:
        weights = np.ones_like(x)

    func_median = lambda y, w: quantile(y, 0.5, weights=w)
    func_mean = lambda y, w: np.average(y, weights=w).reshape(1)

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

    ax.plot(loads[:, 0], loads[:, 1], **plot_kwargs)

    if ifscatter:
        if fkind == "errorbar":
            ax.errorbar(loads[:, 0],
                        loads[:, 1],
                        yerr=(loads[:, 2] - loads[:, 3]) / 2.0,
                        **plot_scatter_kwargs)
        elif fkind == "fbetween":
            ax.fill_between(loads[:, 0], loads[:, 3], loads[:, 2],
                            **plot_scatter_kwargs)



def plot_scatter(x,
                 y,
                 fig=None,
                 ax=None,
                 range=None,
                 auto_p=None,
                 weights=None,
                 plot_kwargs=None):

    bad = flag_bad(x) | flag_bad(y)
    x = x[~bad]
    y = y[~bad]

    if range is None:
        xrange = [x.min(), x.max()]
        yrange = [y.min(), y.max()]
    elif range == 'auto':
        if auto_p is None:
            auto_p = ([1, 99], [1, 99])
        xrange = [
            np.percentile(x, auto_p[0][0]),
            np.percentile(x, auto_p[0][1])
        ]
        yrange = [
            np.percentile(y, auto_p[1][0]),
            np.percentile(y, auto_p[1][1])
        ]
    else:
        xrange = range[:][0]
        yrange = range[:][1]

    is_in_range = (x > xrange[0]) & (x < xrange[1]) & (y > yrange[0]) & (y < yrange[1])
    if weights is None:
        weights = np.ones_like(x)

    ax.scatter(x[is_in_range], y[is_in_range], **plot_kwargs)



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
