import itertools
import re
from typing import Union, List, Callable, Any, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic, binned_statistic_2d
from .plot_methods import plot_contour, plot_trend, plot_corner, plot_scatter, plot_heatmap
from .utils import string_to_list, is_string_or_list_of_string, list_reshape, flag_bad

_range = range

# TODO: DF to AASTeX tabel. Maybe ref to: https://github.com/liuguanfu1120/Excel-to-AASTeX/blob/main/xlsx-to-AAS-table.ipynb


class BasicDataset:

    # TODO: auto complete for self['x']

    OP_MAP: Dict[str, Callable] = {'log10': np.log10, 'square': np.square}
    OP_MAP_LABEL: Dict[str, str] = {'log10': r'$\log$', 'square': ''}

    def __init__(self,
                 data,
                 names=None,
                 labels: Union[Dict, List, None] = None,
                 ranges: Union[Dict, List, None] = None,
                 unit_labels: Union[Dict, List, None] = None) -> None:
        # TODO: units

        self.data: pd.DataFrame
        self.names: np.ndarray
        self.labels: Dict[str, str]
        self.ranges: Dict[str, Union[List, None]]
        self.unit_labels: Dict[str, str]

        if isinstance(data, pd.DataFrame):
            self.data = data
            if names is None:
                names = data.columns
            else:
                self.data.columns = names
        else:
            self.data = pd.DataFrame(data, columns=names)
            if names is None:
                names = [f'x{i}' for i in range(data.shape[1])]
                self.data.columns = names

        self.names = np.asarray(names, dtype='<U64')

        if isinstance(labels, dict):
            self.labels = labels
        elif isinstance(labels, list):
            self.labels = {name: labels[i] for i, name in enumerate(names)}
        elif labels is None:
            self.labels = {}
        else:
            raise ValueError('labels should be dict or list')

        if isinstance(ranges, dict):
            self.ranges = ranges
        elif isinstance(ranges, list):
            self.ranges = {name: ranges[i] for i, name in enumerate(names)}
        elif ranges is None:
            self.ranges = {}
        else:
            raise ValueError('ranges should be dict or list')

        if isinstance(unit_labels, dict):
            self.unit_labels = unit_labels
        elif isinstance(unit_labels, list):
            self.unit_labels = {
                name: unit_labels[i]
                for i, name in enumerate(names)
            }
        elif unit_labels is None:
            self.unit_labels = {}
        else:
            raise ValueError('unit_labels should be dict or list')

    def __iter__(self):
        return iter(self.data.columns)

    def __contains__(self, key):
        return key in self.data.columns

    def __getitem__(self, key) -> Union[pd.DataFrame, pd.Series]:
        '''
        -- NOTE -- Should return DataFrame or Series
        '''

        if isinstance(key, tuple):
            if len(key) != 2:
                raise ValueError('key should be a tuple of length 2')
            if is_string_or_list_of_string(key[0]):
                raise ValueError('key[0] can not be string or list of string')
            if is_string_or_list_of_string(key[1]):
                k1_idx, k2 = key
                names_list = list(self.names)
                if isinstance(k2, str):
                    k2_idx: Union[int, List[int]] = names_list.index(k2)
                else:
                    k2_idx = [names_list.index(this_k2) for this_k2 in k2]
            else:
                k1_idx, k2_idx = key
            return self.data.iloc[k1_idx, k2_idx]
        elif is_string_or_list_of_string(key):
            names_list = list(self.names)
            if isinstance(key, str):
                key_idx: Union[int, List[int]] = names_list.index(key)
            else:
                key_idx = [names_list.index(this_k) for this_k in key]
            return self.data.iloc[:, key_idx]
        else:
            return self.data.iloc[key]

    def __setitem__(self, key, value) -> None:

        k2_idx: Union[int, List[int]]

        if isinstance(key, tuple):
            if len(key) != 2:
                raise ValueError('key should be a tuple of length 2')
            if is_string_or_list_of_string(key[0]):
                raise ValueError('key[0] can not be string or list of string')
            if is_string_or_list_of_string(key[1]):
                k1_idx, k2 = key
                names_list = list(self.names)
                if isinstance(k2, str):
                    k2_idx = names_list.index(k2)
                else:
                    k2_idx = [names_list.index(this_k2) for this_k2 in k2]
            else:
                k1_idx, k2_idx = key
            self.data.iloc[k1_idx, k2_idx] = value

        elif is_string_or_list_of_string(key):
            names_list = list(self.names)

            if isinstance(key, str):
                if key in names_list:
                    key_idx: Union[int, List[int],
                                   None] = names_list.index(key)
                else:
                    self.add_col(value, key)
                    key_idx = None
            else:
                key_idx = []
                new_names = []
                for this_k in key:
                    if this_k in names_list:
                        key_idx.append(names_list.index(this_k))
                    else:
                        new_names.append(this_k)

                # sourcery skip: simplify-len-comparison
                if len(new_names) > 0:
                    self.add_col(value, new_names)

            if key_idx is not None:
                self.data.iloc[:, key_idx] = value

        else:
            self.data.iloc[key] = value

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        summary_string = "" + 'Dataset summary:\n'
        summary_string += f'  Data shape: {str(self.data.shape)}' + '\n'
        summary_string += f'  Names: {str(self.names)}' + '\n'
        label_lst = [self.labels.get(name, name) for name in self.names]
        summary_string += f'  Labels: {str(label_lst)}' + '\n'
        return summary_string

    def update_labels(self, labels_dict) -> None:
        self.labels.update(labels_dict)

    def update_unit_labels(self, unit_labels_dict) -> None:
        self.unit_labels.update(unit_labels_dict)

    def update_names(self, names_dict) -> None:
        for name in names_dict:
            idx = self.names == name
            self.names[idx] = names_dict[name]
            self.data.rename(columns={name: names_dict[name]}, inplace=True)

    def update_ranges(self, ranges_dict) -> None:
        self.ranges.update(ranges_dict)

    def summary(self, stats_info=False) -> None:
        print(self.__str__())
        if stats_info:
            print(self.data.describe())

    def add_col(self, new_cols, new_names) -> None:

        new_names = string_to_list(new_names)

        for name in new_names:
            if name in self.names:
                raise ValueError(f'{name} already exists in the dataset')

        new_cols = np.asarray(new_cols)
        if new_cols.ndim == 1:
            new_cols = new_cols[:, np.newaxis]

        # self.data is a DataFrame
        self.data = pd.concat(
            [self.data, pd.DataFrame(new_cols, columns=new_names)], axis=1)
        self.names = np.asarray(list(self.names) + list(new_names))

    def add_row(self, new_rows) -> None:
        self.data = pd.concat(
            [self.data, pd.DataFrame(new_rows, columns=self.names)],
            axis=0,
            ignore_index=True)

    def del_col(self, key) -> None:
        '''
        deleta the data by index or name.
        '''
        if is_string_or_list_of_string(key):
            names_list = list(self.names)
            if isinstance(key, str):
                key_idx: Union[int, List[int]] = names_list.index(key)
            else:
                key_idx = [names_list.index(this_k) for this_k in key]
            self._del_col(key_idx)
        else:
            self._del_col(key)

    def _del_col(self, key):
        # self.data in a Pandas DataFrame
        self.data.drop(self.names[key], axis=1, inplace=True)
        self.names = np.delete(self.names, key, axis=0)

    def del_row(self, nrow) -> None:
        self.data.drop(nrow, axis=0, inplace=True)
        # reindex
        self.data.reset_index(drop=True, inplace=True)

    def get_data_by_name(self, name) -> np.ndarray:
        # sourcery skip: remove-unnecessary-else, swap-if-else-branches
        if '@' in name:
            op, name = name.split('@')
            return self.OP_MAP[op](self[name].to_numpy())
        else:
            return self[name].to_numpy()

    def get_data_by_names(self, names) -> np.ndarray:
        return np.asarray([self.get_data_by_name(name) for name in names]).T

    def get_label_by_name(self, name, with_unit=True) -> str:
        # sourcery skip: remove-unnecessary-else, swap-if-else-branches

        if with_unit:
            unit = self.get_unit_label_by_name(name)
        else:
            unit = ''

        if '@' in name:
            op, name = name.split('@')
            return self.OP_MAP_LABEL[op] + self.labels.get(name, name) + unit
        else:
            return self.labels.get(name, name) + unit

    def get_labels_by_names(self, names, with_unit=True) -> List[str]:
        return [
            self.get_label_by_name(name, with_unit=with_unit) for name in names
        ]

    def get_unit_label_by_name(self, name):
        if '@' in name:
            _, name = name.split('@')
            unit = ' ' + self.unit_labels.get(name, '')
        else:
            unit = ' ' + self.unit_labels.get(name, '')
        return unit

    def get_range_by_name(self, name):
        if '@' in name:
            if name in self.ranges:
                return self.ranges[name]
            else:
                op, name = name.split('@')
                range_original = self.ranges.get(name, None)
                if range_original is None:
                    return None
                else:
                    range_min = self.OP_MAP[op](range_original[0])
                    range_max = self.OP_MAP[op](range_original[1])
                    if flag_bad(range_min) or flag_bad(range_max):
                        return None
                    return [
                        self.OP_MAP[op](range_original[0]),
                        self.OP_MAP[op](range_original[1])
                    ]
        else:
            return self.ranges.get(name, None)

    def get_subsample(
        self, subsample: Union[None, str, np.ndarray]
    ) -> np.ndarray:  # sourcery skip: lift-return-into-if
        _subsample: np.ndarray

        if subsample is None:
            _subsample = np.ones(self.data.shape[0]).astype(bool)
        elif isinstance(subsample, str):
            _subsample = self.string_to_subsample(subsample)
        else:
            _subsample = subsample

        if _subsample.dtype == int:
            _subsample = self.index_to_bool_subsample(_subsample)

        return _subsample

    def index_to_bool_subsample(self, index) -> np.ndarray:
        _subsample = np.zeros(self.data.shape[0]).astype(bool)
        _subsample[index] = True
        return _subsample

    def string_to_subsample(self, string) -> np.ndarray:
        # sourcery skip: lift-return-into-if, remove-unnecessary-else

        if is_inequality(string):
            _subsample = self.inequality_to_subsample(string)
        else:
            names_list = list(self.names)
            subsample_idx = names_list.index(string)
            _subsample = self[:, subsample_idx].astype(bool).to_numpy()
        return _subsample

    def inequality_to_subsample(self,
                                inequality_string,
                                debug=False) -> np.ndarray:
        '''
        Return the subsample according to the inequality string.
        '''
        # TODO: support & and |
        inequality_list = parse_inequality(inequality_string)
        names_list = list(self.names)
        subsample = np.ones(self.data.shape[0]).astype(bool)

        op_list = ['<=', '>=', '<', '>']
        for i, string in enumerate(inequality_list[2:]):
            if string not in op_list:
                this_inequality = inequality_list[i:i + 3]
                for j in range(len(this_inequality)):
                    if this_inequality[j] in names_list:
                        this_inequality[j] = f"self['{this_inequality[j]}']"

                command = "".join(this_inequality)
                if debug:
                    print(this_inequality)

                subsample = subsample & eval(command).to_numpy()

        return subsample

    def get_subsample_each_bin_by_name(
        self,
        names,
        bins=10,
        title_ndigits=2,
        return_edges=False,
        list_shape=None,
        range=None,
        subsample=None
    ) -> Union[tuple[List, List], tuple[List, List, Union[List, np.ndarray]]]:
        """
        list_reshape: 
            return subsample_each and title_each in shape defined by list_reshape. Only works when use one name.
        """

        title_each: List = []
        subsample_each: List = []

        names = string_to_list(names)
        subsample = self.get_subsample(subsample)

        if len(names) == 1:
            x = self.get_data_by_name(names[0])
            _, edges, bin_index = binned_statistic(x,
                                                   x,
                                                   statistic='count',
                                                   bins=bins,
                                                   range=range)
            for i in _range(1, len(edges)):

                # TODO: consider log10@x
                # TODO: different format
                # TODO: drop min/max for first/latest bin
                title_each.append(
                    f'{names[0]}: [{edges[i-1]:.{title_ndigits}f}, {edges[i]:.{title_ndigits}f})'
                )

                subsample_each.append(subsample & (bin_index == i))

            if list_shape is not None:
                title_each = list_reshape(title_each, list_shape)
                subsample_each = list_reshape(subsample_each, list_shape)

        elif len(names) == 2:
            x = self.get_data_by_name(names[0])
            y = self.get_data_by_name(names[1])
            _, x_edges, y_edges, bin_index = binned_statistic_2d(
                x,
                y,
                x,
                statistic='count',
                bins=bins,
                range=range,
                expand_binnumbers=True)

            for i in _range(1, len(x_edges)):
                for j in _range(1, len(y_edges)):
                    title_each.append(
                        f'{names[0]}: [{x_edges[i-1]:.{title_ndigits}f}, {x_edges[i]:.{title_ndigits}f}), {names[1]}: [{y_edges[j-1]:.{title_ndigits}f}, {y_edges[j]:.{title_ndigits}f})'
                    )
                    subsample_each.append(subsample & (bin_index[0] == i)
                                          & (bin_index[1] == j))

            edges = [x_edges, y_edges]
        else:
            raise ValueError('can not handle more than two names')

        if return_edges:
            return subsample_each, title_each, edges
        return subsample_each, title_each


class Dataset(BasicDataset):

    # -- Note -- that all values passed to plot_xxx should be numpy array, not series

    # TODO: histogram
    # TODO: control 1D/2D

    def __init__(self,
                 data,
                 names=None,
                 labels=None,
                 ranges=None,
                 unit_labels=None) -> None:

        super().__init__(data,
                         names=names,
                         labels=labels,
                         ranges=ranges,
                         unit_labels=unit_labels)

        self.method_mapping = {
            'trend': self._trend,
            'contour': self._contour,
            'scatter': self._scatter,
            'heatmap': self._heatmap
        }

    def _trend(self,
               x_name,
               y_name,
               ax,
               subsample=None,
               xlabel=None,
               ylabel=None,
               title=None,
               xlim=None,
               ylim=None,
               **kwargs):

        x = self.get_data_by_name(x_name)
        y = self.get_data_by_name(y_name)
        _subsample = self.get_subsample(subsample)
        weights = kwargs.pop('weights', None)
        weights = self.get_data_by_name(weights) if isinstance(
            weights, str) else weights
        _weights = weights[_subsample] if weights is not None else None
        plot_trend(x[_subsample],
                   y[_subsample],
                   ax=ax,
                   weights=_weights,
                   **kwargs)
        self._set_ax_prperties(ax, x_name, y_name, xlabel, ylabel, title, xlim,
                               ylim)
        ax.legend()

    def _heatmap(self,
                 x_name,
                 y_name,
                 ax,
                 z_name=None,
                 subsample=None,
                 xlabel=None,
                 ylabel=None,
                 title=None,
                 xlim=None,
                 ylim=None,
                 **kwargs):

        x = self.get_data_by_name(x_name)
        y = self.get_data_by_name(y_name)

        if z_name is None:
            z_name = np.ones_like(x)
            print("z_name is not specified, use np.ones_like(x) instead")
            print("I think you'd like to specify z_name")

        z = self.get_data_by_name(z_name)

        _subsample = self.get_subsample(subsample)
        weights = kwargs.pop('weights', None)
        weights = self.get_data_by_name(weights) if isinstance(
            weights, str) else weights
        _weights = weights[_subsample] if weights is not None else None
        plot_heatmap(x[_subsample],
                     y[_subsample],
                     z[_subsample],
                     ax=ax,
                     weights=_weights,
                     **kwargs)

        self._set_ax_prperties(ax, x_name, y_name, xlabel, ylabel, title, xlim,
                               ylim)

    def _contour(self,
                 x_name,
                 y_name,
                 ax,
                 subsample=None,
                 xlabel=None,
                 ylabel=None,
                 title=None,
                 xlim=None,
                 ylim=None,
                 **kwargs):
        '''
        xlabel:
            If False, do not set xlabel
            If None, set from self.labels
            If string, set as xlabel
        
        ylabel:
            If False, do not set ylabel
            If None, set from self.labels
            If string, set as ylabel

        title:
            If None or False, do not set title
            If string, set as title

        xlim:
            If None, do not set xlim
            If list, set as xlim

        ylim:
            If None, do not set ylim
            If list, set as ylim

        '''

        x = self.get_data_by_name(x_name)
        y = self.get_data_by_name(y_name)

        _subsample = self.get_subsample(subsample)

        weights = kwargs.pop('weights', None)
        weights = self.get_data_by_name(weights) if isinstance(
            weights, str) else weights
        _weights = weights[_subsample] if weights is not None else None

        if not 'range' in kwargs:
            kwargs['range'] = self._get_default_range(x_name, y_name)

        plot_contour(x[_subsample],
                     y[_subsample],
                     ax=ax,
                     weights=_weights,
                     **kwargs)

        self._set_ax_prperties(ax, x_name, y_name, xlabel, ylabel, title, xlim,
                               ylim)

    def _get_default_range(self, x_name, y_name):
        xrange = self.get_range_by_name(x_name)
        if xrange is None:
            x = self.get_data_by_name(x_name)
            xrange = [x.min(), x.max()]
        yrange = self.get_range_by_name(y_name)
        if yrange is None:
            y = self.get_data_by_name(y_name)
            yrange = [y.min(), y.max()]
        return [xrange, yrange]

    def _set_ax_prperties(self, ax, x_name, y_name, xlabel, ylabel, title,
                          xlim, ylim):
        if (title is not False) and (title is not None):
            ax.set_title(title)

        if xlabel is not False:
            if xlabel is None:
                xlabel = self.get_label_by_name(x_name)
            ax.set_xlabel(xlabel)

        if ylabel is not False:
            if ylabel is None:
                ylabel = self.get_label_by_name(y_name)
            ax.set_ylabel(ylabel)

        if xlim is not None:
            ax.set_xlim(xlim)

        if ylim is not None:
            ax.set_ylim(ylim)

    def _scatter(self,
                 x_name,
                 y_name,
                 ax,
                 z_name=None,
                 subsample=None,
                 xlabel=None,
                 ylabel=None,
                 title=None,
                 xlim=None,
                 ylim=None,
                 **kwargs):

        x = self.get_data_by_name(x_name)
        y = self.get_data_by_name(y_name)
        if z_name is None:
            _z = None
        else:
            _z = self.get_data_by_name(z_name)

        _subsample = self.get_subsample(subsample)
        weights = kwargs.pop('weights', None)
        weights = self.get_data_by_name(weights) if isinstance(
            weights, str) else weights
        _weights = weights[_subsample] if weights is not None else None
        plot_scatter(x[_subsample],
                     y[_subsample],
                     z=_z,
                     ax=ax,
                     weights=_weights,
                     **kwargs)
        self._set_ax_prperties(ax, x_name, y_name, xlabel, ylabel, title, xlim,
                               ylim)
        if kwargs.get('label', None) is not None:
            ax.legend()

    def plot_xygeneral_no_broadcast(self,
                                    kind,
                                    x_names,
                                    y_names,
                                    axes=None,
                                    subplots_kwargs=None,
                                    **kwargs):
        x_names = string_to_list(x_names)
        y_names = string_to_list(y_names)

        if len(x_names) != len(y_names):
            raise ValueError('x_names and y_names have different length')

        if subplots_kwargs is None:
            subplots_kwargs = {}

        if axes is None:
            fig, axes = auto_subplots(len(x_names), **subplots_kwargs)

        # If axes is a single ax, convert it to an array
        if not hasattr(axes, '__iter__'):
            axes = np.array([axes])

        # find fig by axes
        fig = axes.flatten()[0].get_figure()

        same_key = {}
        each_key = {}
        for key in kwargs:
            # is end of
            if key.endswith('_each'):
                key_single = key[:-5]
                each_key[key_single] = kwargs[key]
            else:
                same_key[key] = kwargs[key]

        for i in range(len(x_names)):
            ax = axes.flat[i]
            this_kwargs = same_key.copy()
            for key in each_key:
                this_kwargs[key] = each_key[key][i]
            self.method_mapping[kind](x_names[i], y_names[i], ax,
                                      **this_kwargs)

        return fig, axes

    def plot_xygeneral(self,
                       kind,
                       x_names,
                       y_names,
                       axes=None,
                       subplots_kwargs=None,
                       **kwargs):

        # TODO: bin by the third variable

        x_names = string_to_list(x_names)
        y_names = string_to_list(y_names)

        n1 = len(x_names)
        n2 = len(y_names)

        if (n1 > 1) and (n2 > 1):
            scatter_type = 'xy'
        elif (n1 > 1) and (n2 == 1):
            scatter_type = 'x'
        elif (n1 == 1) and (n2 > 1):
            scatter_type = 'y'
        elif (n1 == 1) and (n2 == 1):
            scatter_type = 'single'

        if subplots_kwargs is None:
            subplots_kwargs = {}

        if axes is None:
            if scatter_type == 'xy':
                fig, axes = auto_subplots(n1, n2, **subplots_kwargs)
            else:
                fig, axes = auto_subplots(n1 * n2, **subplots_kwargs)

        # If axes is a single ax, convert it to an array
        if not hasattr(axes, '__iter__'):
            axes = np.array([axes])

        # find fig by axes
        fig = axes.flatten()[0].get_figure()

        same_key = {}
        each_key = {}
        for key in kwargs:
            # is end of
            if key.endswith('_each'):
                key_single = key[:-5]
                each_key[key_single] = kwargs[key]
                # If it is 1D list, convert it to 2D list
                '''
                Note that this code can not solve all problems
                Corner case: range = [[1, 2], [1, 2]]
                No plan to deal with such corner case
                If anyone has a good idea, please let me know
                '''
                if not isinstance(each_key[key_single][0], list):
                    each_key[key_single] = [each_key[key_single]]
            else:
                same_key[key] = kwargs[key]

        # i, vertical, y; j, horizontal, x
        for (i, j), ax in zip(itertools.product(range(n2), range(n1)),
                              axes.flatten()):
            this_kwargs = same_key.copy()

            for key in each_key:
                if scatter_type == 'xy':
                    this_kwargs[key] = each_key[key][i][j]
                elif scatter_type in ['x', 'single']:
                    this_kwargs[key] = each_key[key][0][j]
                elif scatter_type == 'y':
                    this_kwargs[key] = each_key[key][0][i]

            self.method_mapping[kind](x_names[j], y_names[i], ax,
                                      **this_kwargs)

        return fig, axes

    def trend(self,
              x_names,
              y_names,
              broadcast=True,
              axes=None,
              subplots_kwargs=None,
              **kwargs):

        if broadcast:
            return self.plot_xygeneral('trend',
                                       x_names,
                                       y_names,
                                       axes=axes,
                                       subplots_kwargs=subplots_kwargs,
                                       **kwargs)

        else:
            return self.plot_xygeneral_no_broadcast(
                'trend',
                x_names,
                y_names,
                axes=axes,
                subplots_kwargs=subplots_kwargs,
                **kwargs)

    def contour(self,
                x_names,
                y_names,
                broadcast=True,
                axes=None,
                subplots_kwargs=None,
                **kwargs):

        if broadcast:
            return self.plot_xygeneral('contour',
                                       x_names,
                                       y_names,
                                       axes=axes,
                                       subplots_kwargs=subplots_kwargs,
                                       **kwargs)
        else:
            return self.plot_xygeneral_no_broadcast(
                'contour',
                x_names,
                y_names,
                axes=axes,
                subplots_kwargs=subplots_kwargs,
                **kwargs)

    def corner(self, names=None, axes=None, **kwargs):
        if names is None:
            names = self.names

        if 'labels' not in kwargs:
            kwargs['labels'] = [self.get_label_by_name(name) for name in names]

        if axes is not None:
            axes = np.atleast_1d(axes)
            fig = axes.flatten()[0].get_figure()
        else:
            fig = None

        xs = np.array([self.get_data_by_name(name) for name in names]).T
        return plot_corner(xs, fig=fig, **kwargs)

    def scatter(self,
                x_names,
                y_names,
                broadcast=True,
                axes=None,
                subplots_kwargs=None,
                **kwargs):

        if broadcast:
            return self.plot_xygeneral('scatter',
                                       x_names,
                                       y_names,
                                       axes=axes,
                                       subplots_kwargs=subplots_kwargs,
                                       **kwargs)
        else:
            return self.plot_xygeneral_no_broadcast(
                'scatter',
                x_names,
                y_names,
                axes=axes,
                subplots_kwargs=subplots_kwargs,
                **kwargs)

    def heatmap(self,
                x_names,
                y_names,
                broadcast=True,
                axes=None,
                subplots_kwargs=None,
                **kwargs):

        if broadcast:
            return self.plot_xygeneral('heatmap',
                                       x_names,
                                       y_names,
                                       axes=axes,
                                       subplots_kwargs=subplots_kwargs,
                                       **kwargs)
        else:
            return self.plot_xygeneral_no_broadcast(
                'heatmap',
                x_names,
                y_names,
                axes=axes,
                subplots_kwargs=subplots_kwargs,
                **kwargs)

    def get_RF_importance(self,
                          x_names,
                          y_name,
                          problem_type=None,
                          subsample=None,
                          **kwargs):
        ...
        x_names = string_to_list(x_names)
        xs = self.get_data_by_names(x_names)
        y = self.get_data_by_name(y_name)
        _subsample = self.get_subsample(subsample)
        xs = xs[_subsample]
        y = y[_subsample]

        if problem_type is None:
            print('problem_type is not specified, try to guess:')
            print('  If y is float, problem_type is regression')
            print('  If y is int or bool, problem_type is classification')
            if is_float(y):
                problem_type = 'regression'
            elif is_int(y) or is_bool(y):
                problem_type = 'classification'
            else:
                raise ValueError(
                    'Can not guess problem_type, please specify problem_type')

        # return get_RF_importance(self.get_data_by_name(x_names), y[_subsample], weights=_weights, **kwargs)


def auto_subplots(n1, n2=None, figshape=None, figsize=None, dpi=400):
    if figshape is None:
        if n2 is None:
            figshape = (int(np.ceil(np.sqrt(n1))), int(np.ceil(np.sqrt(n1))))
        else:
            figshape = (n2, n1)  # vertical, horizontal
    if figsize is None:
        figsize = (figshape[1] * 4, figshape[0] * 4)
    fig, axes = plt.subplots(figshape[0],
                             figshape[1],
                             figsize=figsize,
                             dpi=400)
    if not hasattr(axes, '__iter__'):
        axes = np.array([axes])
    return fig, axes


def parse_inequality(inequaliyt_string):
    return re.split(r'(<=|>=|<|>)', inequaliyt_string.replace(" ", ""))


def is_inequality(string):
    return re.search(r'(<=|>=|<|>)', string) is not None
