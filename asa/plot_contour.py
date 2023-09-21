from Bcorner import hist2d


def plot_contour(x,
                 y,
                 bins=20,
                 range='auto',
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

