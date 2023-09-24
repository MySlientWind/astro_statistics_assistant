import numpy as np
import matplotlib.pyplot as plt
from .plot_contour import plot_contour
from .plot_trend import plot_trend


class Dataset:
    def __init__(self, data, names, labels) -> None:
        self.data = np.asarray(data)
        self.names = np.asarray(names)
        self.labels = np.asarray(labels)

    def add_col(self, new_cols, new_names, new_labels) -> None:

        new_cols = np.asarray(new_cols)
        if new_cols.ndim == 1:
            new_cols = new_cols[:, np.newaxis]
        new_names = string_to_list(new_names)
        new_labels = string_to_list(new_labels)

        self.data = np.hstack((self.data, new_cols))
        self.names = np.asarray(list(self.names) + list(new_names))
        self.labels = np.asarray(list(self.labels) + list(new_labels))

    def add_row(self, new_rows) -> None:
        self.data = np.vstack((self.data, new_rows))

    def _trend(self, x_name, y_name, ax, **kwargs):

        # TODO: scatter, etc.

        names_list = list(self.names)
        x_idx = names_list.index(x_name)
        y_idx = names_list.index(y_name)
        plot_trend(self.data[:, x_idx], self.data[:, y_idx], ax=ax, **kwargs)
        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)

    def trend(self,
                x_name,
                y_names,
                axes=None,
                subplots_kwargs=None,
                **kwargs):

        # TODO: kwargs_each to use different kwargs for each plot

        # if y_names is a string
        y_names = string_to_list(y_names)

        if subplots_kwargs is None:
            subplots_kwargs = {}
       
        if axes is None:
            _, axes = auto_subplots(len(y_names), **subplots_kwargs)

        for y_name, ax in zip(y_names, axes.flatten()):
            self._trend(x_name, y_name, ax, **kwargs)


    def _contour(self, x_name, y_name, ax, **kwargs):

        # TODO: labels, tiles, ranges, etc.

        names_list = list(self.names)
        x_idx = names_list.index(x_name)
        y_idx = names_list.index(y_name)
        plot_contour(self.data[:, x_idx], self.data[:, y_idx], ax=ax, **kwargs)
        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)

    def contour(self,
                x_name,
                y_names,
                axes=None,
                subplots_kwargs=None,
                **kwargs):

        # TODO: kwargs_each to use different kwargs for each plot
        # TODO: contour plot bin by the third variable

        # if y_names is a string
        y_names = string_to_list(y_names)

        if subplots_kwargs is None:
            subplots_kwargs = {}

        if axes is None:
            _, axes = auto_subplots(len(y_names), **subplots_kwargs)

        for y_name, ax in zip(y_names, axes.flatten()):
            self._contour(x_name, y_name, ax, **kwargs)


def auto_subplots(n, figshape=None, figsize=None, dpi=400):
    if figshape is None:
        figshape = (int(np.ceil(np.sqrt(n))), int(np.ceil(np.sqrt(n))))
    if figsize is None:
        figsize = (figshape[1] * 4, figshape[0] * 4)
    fig, axes = plt.subplots(figshape[0],
                             figshape[1],
                             figsize=figsize,
                             dpi=400)
    if n == 1:
        axes = np.array([axes])
    return fig, axes

def string_to_list(string):
    return [string] if isinstance(string, str) else string