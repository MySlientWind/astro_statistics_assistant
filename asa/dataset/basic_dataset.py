from typing import Callable, Dict, List, Union

import astropy.units as u
import numpy as np
import pandas as pd

from .. import uncertainty as unc
from ..binning_methods import (binned_statistic_2d_robust,
                               binned_statistic_robust)
from ..utils import (flag_bad, is_string_or_list_of_string, list_reshape,
                     string_to_list)
from .inequality_utlis import parse_and_or, parse_inequality, parse_op, is_inequality
from .labels import OP_MAP_LABEL, _get_label_by_name

_range = range


class BasicDataset:

    # TODO: auto complete for self['x']
    # TODO: DF to AASTeX tabel. Maybe ref to: https://github.com/liuguanfu1120/Excel-to-AASTeX/blob/main/xlsx-to-AAS-table.ipynb
    # TODO: better OP_MAP_LABEL with format string.
    # TODO: dict as input
    # TODO: check if - in name, we require the key should be a legal var name

    OP_MAP: Dict[str, Callable] = {'log10': np.log10, 'square': np.square}
    OP_MAP_LABEL: Dict[str, str] = OP_MAP_LABEL
    OP_SNR_MAP: Dict[str, Callable] = {
        'log10': unc.log10_snr,
        'square': unc.square_snr
    }
    OP_ERR_MAP: Dict[str, Callable] = {
        'log10': unc.log10,
        'square': unc.square
    }

    def __init__(self,
                 data,
                 names=None,
                 labels: Union[Dict, List, None] = None,
                 ranges: Union[Dict, List, None] = None,
                 unit_labels: Union[Dict, List, None] = None,
                 units: Union[Dict, List, None] = None,
                 snr_postfix='snr',
                 err_postfix='err') -> None:

        # self.names: np.ndarray

        self.data: pd.DataFrame
        self.labels: Dict[str, str]
        self.ranges: Dict[str, Union[List, None]]
        self.unit_labels: Dict[str, str]
        self.units: Dict[str, u.Unit]

        self.snr_postfix: str = snr_postfix
        self.err_postfix: str = err_postfix

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

        self.labels = self._init_parameter(labels, 'labels', names)
        self.ranges = self._init_parameter(ranges, 'ranges', names)
        self.unit_labels = self._init_parameter(unit_labels, 'unit_labels',
                                                names)
        self.units = self._init_parameter(units, 'units', names)

    def _init_parameter(self, parameter, parameter_name, names):

        if isinstance(parameter, dict):
            return parameter
        elif isinstance(parameter, list):
            return {name: parameter[i] for i, name in enumerate(names)}
        elif parameter is None:
            return {}
        else:
            raise ValueError(f'{parameter_name} should be dict or list')

    @property
    def names(self):
        return np.asarray(self.data.columns, dtype='<U64')

    @property
    def shape(self):
        return self.data.shape

    def __iter__(self):
        return iter(self.data.columns)

    def __contains__(self, key):
        return key in self.data.columns

    def __getitem__(self, key) -> Union[pd.DataFrame, pd.Series]:

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
                    self.data.iloc[:, names_list.index(key)] = value
                else:
                    self.add_col(value, key)
            else:
                key_idx = []
                new_names = []
                new_value_idx = []
                old_value_idx = []
                for i, this_k in enumerate(key):
                    if this_k in names_list:
                        key_idx.append(names_list.index(this_k))
                        old_value_idx.append(i)
                    else:
                        new_names.append(this_k)
                        new_value_idx.append(i)

                # sourcery skip: simplify-len-comparison
                # add new columns
                if len(new_names) > 0:
                    self.add_col(value[:, new_value_idx], new_names)

                # update existing columns
                if key_idx:
                    self.data.iloc[:, key_idx] = value[:, old_value_idx]

        else:
            self.data.iloc[key] = value

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        summary_string = "" + 'Dataset summary:\n'
        summary_string += f'  Data shape: {self.shape}\n'
        summary_string += f'  Names: {self.names}\n'
        label_lst = [self.labels.get(name, name) for name in self.names]
        summary_string += f'  Labels: {label_lst}\n'
        return summary_string

    def update_labels(self, labels_dict) -> None:
        self.labels.update(labels_dict)

    def update_unit_labels(self, unit_labels_dict) -> None:
        self.unit_labels.update(unit_labels_dict)

    def update_names(self, names_dict) -> None:
        self.data.rename(columns=names_dict, inplace=True)

    def update_ranges(self, ranges_dict) -> None:
        self.ranges.update(ranges_dict)

    def update_units(self, units_dict) -> None:
        self.units.update(units_dict)

    def summary(self, stats_info=False) -> None:
        print(str(self))
        if stats_info:
            print(self.data.describe())

    def add_col(self, new_cols, new_names) -> None:
        '''
        Input:
            new_cols: scaler, array-like, or list (or tuple) of scaler or array-like
                      if list (or tuple) with length (n_features, ), each element should be array-like. If want to add multiple columns with unit, can only use this 
                      format.
                      
            new_names: str or list of str, (n_features, )

        Add new columns to the dataset
        '''

        new_names = string_to_list(new_names)

        for name in new_names:
            if name in self.names:
                raise ValueError(f'{name} already exists in the dataset')

        n_names = len(new_names)

        if isinstance(new_cols, list) or isinstance(new_cols, tuple):
            if n_names == len(new_cols):
                for nc, nn in zip(new_cols, new_names):
                    self.add_col(nc, nn)
                return

        # should use asanyarray to handle Quantity
        new_cols = np.asanyarray(new_cols)

        if new_cols.ndim == 0:
            if n_names > 1:
                raise ValueError(
                    f"new_cols can only be a scalar if new_names has length 1, but got {n_names} \n consider use list (or tuple) of scaler instead."
                )
        elif new_cols.ndim == 1:
            if n_names > 1:
                raise ValueError(f"1 columns, but {n_names} names")
            if new_cols.shape[0] != self.shape[0]:
                raise ValueError(
                    f"new_cols has length {new_cols.shape[0]}, but the dataset has {self.shape[0]} rows"
                )
        elif new_cols.ndim == 2:
            if new_cols.shape[1] != n_names:
                raise ValueError(
                    f"{new_cols.shape[1]} columns, but {n_names} names")
            if new_cols.shape[0] != self.shape[0]:
                raise ValueError(
                    f"new_cols has {new_cols.shape[0]} rows, but the dataset has {self.shape[0]} rows"
                )
        else:
            raise ValueError(f"Unexpected ndim of new_cols: {new_cols.ndim}")

        if isinstance(new_cols, u.Quantity):

            if len(new_names) > 1:
                raise ValueError(
                    'Multiple columns with unit can only be added by using new_cols as a list (or tuple) of array-like or scaler with length (n_features, )'
                )

            if new_names[0] in self.units:
                new_cols = new_cols.to(self.units[new_names[0]]).value
            else:
                raise ValueError(
                    f'You are trying to add a column with unit, but the unit of {new_names[0]} is not specified in the dataset. Please specify the unit of {new_names[0]} first.'
                )

        self.data.loc[:, new_names] = new_cols

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

    def del_row(self, nrow) -> None:
        self.data.drop(nrow, axis=0, inplace=True)
        # reindex
        self.data.reset_index(drop=True, inplace=True)

    def remove_snr_postfix(self, name) -> str:
        if name.endswith(f'_{self.snr_postfix}'):
            return name[:-len(self.snr_postfix) - 1]
        else:
            raise ValueError(f'{name} does not end with _{self.snr_postfix}')

    def remove_err_postfix(self, name) -> str:
        if name.endswith(f'_{self.err_postfix}'):
            return name[:-len(self.err_postfix) - 1]
        else:
            raise ValueError(f'{name} does not end with _{self.err_postfix}')

    def is_legal_name(self, name) -> bool:
        if name.endswith(f'_{self.snr_postfix}'):
            return self.is_legal_name(self.remove_snr_postfix(name))
        elif name.endswith(f'_{self.err_postfix}'):
            return self.is_legal_name(self.remove_err_postfix(name))
        elif '@' in name:
            op, name = name.split('@')
            return self.is_legal_name(name) & (op in self.OP_MAP)
        else:
            return name in self.names

    def get_data_by_name(self,
                         name,
                         subsample=None,
                         with_unit=False) -> np.ndarray:
        '''
        with_unit:
            If True, return the data with unit
            If False, return the data without unit
            Ignored when get snr, err, or with operation
        '''
        # sourcery skip: remove-unnecessary-else, swap-if-else-branches
        if name.endswith(f'_{self.snr_postfix}'):
            data = self.get_snr_by_name(name)
        elif name.endswith(f'_{self.err_postfix}'):
            data = self.get_err_by_name(name)
        elif '@' in name:
            op, name = name.split('@')
            data = self.OP_MAP[op](self[name].to_numpy())
        else:
            if with_unit:
                data = self[name].to_numpy() * self.get_unit_by_name(name)
            else:
                data = self[name].to_numpy()

        subsample = self.get_subsample(subsample)
        return data[subsample]

    gdn = get_data_by_name

    def get_snr_by_name(self, snr_name: str) -> np.ndarray:
        '''
        input:
            snr_name: string
            in format: [{op}@]{data_name}_{snr_postfix}
            [] means optional
        
        return:
            snr: np.ndarray
            The snr of {op}({data_name})
        '''
        if '@' in snr_name:
            op, snr_name = snr_name.split('@')
            data_name = self.remove_snr_postfix(snr_name)
            return self.OP_SNR_MAP[op](self.get_data_by_name(data_name),
                                       self.get_data_by_name(snr_name))
        else:
            # sourcery skip: remove-unnecessary-else
            # if in names, just return it
            if snr_name in self.names:
                return self[snr_name].to_numpy()
            # if not in names, try to find the snr
            else:
                data_name = self.remove_snr_postfix(snr_name)
                err_name = f'{data_name}_{self.err_postfix}'
                if err_name in self.names:
                    return np.abs(self[data_name].to_numpy()
                                  ) / self[err_name].to_numpy()
                else:
                    raise ValueError(
                        f'can not find err_name: {err_name}, nor snr_name: {snr_name}'
                    )

    def get_err_by_name(self, err_name) -> np.ndarray:

        if '@' in err_name:
            op, err_name = err_name.split('@')
            data_name = self.remove_err_postfix(err_name)
            return self.OP_ERR_MAP[op](self.get_data_by_name(data_name),
                                       self.get_data_by_name(err_name))
        else:
            # sourcery skip: remove-unnecessary-else
            # if in names, just return it
            data_name = self.remove_err_postfix(err_name)
            if err_name in self.names:
                return self[err_name].to_numpy()
            # if not in names, try to find the snr
            else:
                snr_name = f'{data_name}_{self.snr_postfix}'
                if snr_name in self.names:
                    return np.abs(self[data_name].to_numpy()
                                  ) / self[snr_name].to_numpy()
                else:
                    raise ValueError(
                        f'can not find err_name: {err_name}, nor snr_name: {snr_name}'
                    )

    def get_data_by_names(self, names, subsample=None) -> np.ndarray:
        return np.asarray([
            self.get_data_by_name(name, subsample=subsample) for name in names
        ]).T

    gdns = get_data_by_names

    def get_label_by_name(self,
                          name,
                          with_unit=True,
                          op_bracket='{label}') -> str:
        # sourcery skip: remove-unnecessary-else, swap-if-else-branches

        unit_label = self.get_unit_label_by_name(name) if with_unit else ''

        return _get_label_by_name(name,
                                  self.labels,
                                  unit_label=unit_label,
                                  op_bracket=op_bracket,
                                  op_map_label=self.OP_MAP_LABEL)

    gln = get_label_by_name

    def get_labels_by_names(self,
                            names,
                            with_unit=True,
                            op_bracket='{label}') -> List[str]:
        return [
            self.get_label_by_name(name,
                                   with_unit=with_unit,
                                   op_bracket=op_bracket) for name in names
        ]

    glns = get_labels_by_names

    def get_unit_by_name(self, name) -> u.Unit:
        return self.units.get(name, 1)

    def get_unit_label_by_name(self, name):
        if '@' in name:
            _, name = name.split('@')
        return self.unit_labels.get(name, '')

    def get_range_by_name(self, name):
        # sourcery skip: last-if-guard, remove-unnecessary-else
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
            self,
            subsample: Union[None, str, np.ndarray],
            debug=False) -> np.ndarray:  # sourcery skip: lift-return-into-if
        _subsample: np.ndarray
        # TODO: 0, 1 to bool

        if subsample is None:
            _subsample = np.ones(self.data.shape[0]).astype(bool)
        elif isinstance(subsample, str):
            _subsample = self.string_to_subsample(subsample, debug=debug)
        # if only include 0 or 1, convert to bool
        # elif np.unique(subsample).tolist() in [[0], [1], [0, 1], [1, 0]]:
        #     _subsample = subsample.astype(bool)
        else:
            _subsample = subsample

        if _subsample.dtype == int:
            _subsample = self.index_to_bool_subsample(_subsample)

        return _subsample

    def index_to_bool_subsample(self, index) -> np.ndarray:
        _subsample = np.zeros(self.data.shape[0]).astype(bool)
        _subsample[index] = True
        return _subsample

    def string_to_subsample(self, string, debug=False) -> np.ndarray:
        # sourcery skip: lift-return-into-if, remove-unnecessary-else

        if not self.is_legal_name(string):
            if is_inequality(string):
                _subsample = self.inequality_to_subsample(string, debug=debug)
            else:
                raise ValueError(f'{string} is not a legal name or inequality')
        else:
            names_list = list(self.names)
            subsample_idx = names_list.index(string)
            _subsample = self[:, subsample_idx].astype(bool).to_numpy()
        return _subsample

    def random_subsample(self,
                         N,
                         as_bool=False,
                         input_subsample=None) -> np.ndarray:
        '''
        N: int or float
            If > 1, the number of samples to be selected
            If < 1, the fraction of samples to be selected
        '''

        input_subsample = self.get_subsample(input_subsample)
        N_input_subsample = input_subsample.sum()

        if N < 1:
            N = int(N * N_input_subsample)

        if N > N_input_subsample:
            raise ValueError(
                'N should not be larger than the number of samples in the subsample'
            )

        subsample = np.random.choice(N_input_subsample, N, replace=False)
        subsample = input_subsample.nonzero()[0][subsample]

        if as_bool:
            subsample = self.index_to_bool_subsample(subsample)
        return subsample

    # TODO: how about the key that do not exist in the dataset?
    def inequality_to_subsample(self,
                                inequality_string,
                                debug=False) -> np.ndarray:
        meta_inequality_list = parse_and_or(inequality_string)
        if debug:
            print("meta_inequality_list:", meta_inequality_list)
        all_subsample = []
        j = 0

        # pylint: disable-next=consider-using-enumerate
        for i in range(len(meta_inequality_list)):
            if meta_inequality_list[i] in ['[', ']']:
                meta_inequality_list[i] = {
                    '[': '(',
                    ']': ')'
                }[meta_inequality_list[i]]
                continue
            if meta_inequality_list[i] not in ['~', '&', '|']:
                if self.is_legal_name(meta_inequality_list[i]):
                    meta_inequality_list[
                        i] = f'self.string_to_subsample("{meta_inequality_list[i]}")'
                else:
                    all_subsample.append(
                        self.inequality_to_subsample_single(
                            meta_inequality_list[i], debug=debug))
                    meta_inequality_list[i] = f'all_subsample[{j}]'
                    j += 1

        command = "".join(meta_inequality_list)
        if debug:
            # print(meta_inequality_list)
            print("command:", command)
        return eval(command)  # pylint: disable=eval-used

    def inequality_to_subsample_single(self,
                                       inequality_string,
                                       debug=False) -> np.ndarray:
        '''
        Return the subsample according to the inequality string.
        '''
        if debug:
            print("inequality_to_subsample_single begin")

        if not is_inequality(inequality_string):
            raise ValueError(f'{inequality_string} is not an inequality')

        inequality_list = parse_inequality(inequality_string)
        subsample = np.ones(self.data.shape[0]).astype(bool)

        if debug:

            print("inequality_list:", inequality_list)

        op_list = ['<=', '>=', '<', '>', '==']
        # a > b > c <=> (a > b) & (b > c)
        # string begin with 2:, but i begin with 0
        for i, string in enumerate(inequality_list[2:]):
            if debug:
                print("string:", string)
                print("op_list:", op_list)
            if string not in op_list:
                this_inequality = inequality_list[i:i + 3]
                # enumerate [a, >, b]
                # pylint: disable-next=consider-using-enumerate
                for j in range(len(this_inequality)):
                    all_element_in_this = parse_op(this_inequality[j])
                    if debug:
                        print("all_element_in_this:", all_element_in_this)
                    # enumerate [a1, +, a2]
                    for k, ele in enumerate(all_element_in_this):
                        if self.is_legal_name(ele):
                            all_element_in_this[
                                k] = f"self.get_data_by_name('{ele}')"
                    this_inequality[j] = "".join(all_element_in_this)

                command = "".join(this_inequality)
                if debug:
                    print("this_inequality:", this_inequality)
                    print("command:", command)

                subsample = subsample & eval(command)  # pylint: disable=eval-used

        if debug:
            print("inequality_to_subsample_single end")

        return subsample

    def get_subsample_each_bin_by_name(
        self,
        names,
        bins=10,
        title_ndigits=2,
        return_edges=False,
        list_shape=None,
        range=None,  # pylint: disable=redefined-builtin
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

            _, edges, bin_index = binned_statistic_robust(x,
                                                          x,
                                                          statistic='count',
                                                          bins=bins,
                                                          range=range)

            for i in _range(1, len(edges)):

                # TODO: different format
                # TODO: drop min/max for first/latest bin
                title_each.append(
                    f'{self.get_label_by_name(names[0])}: [{edges[i-1]:.{title_ndigits}f}, {edges[i]:.{title_ndigits}f})'
                )

                subsample_each.append(subsample & (bin_index == i))

            if list_shape is not None:
                title_each = list_reshape(title_each, list_shape)
                subsample_each = list_reshape(subsample_each, list_shape)

        elif len(names) == 2:
            x = self.get_data_by_name(names[0])
            y = self.get_data_by_name(names[1])

            _, x_edges, y_edges, bin_index = binned_statistic_2d_robust(
                x, y, x, statistic='count', bins=bins, range=range)

            for i in _range(1, len(x_edges)):
                for j in _range(1, len(y_edges)):
                    title_each.append(
                        f'{self.get_label_by_name(names[0])}: [{x_edges[i-1]:.{title_ndigits}f}, {x_edges[i]:.{title_ndigits}f}), {self.get_label_by_name(names[1])}: [{y_edges[j-1]:.{title_ndigits}f}, {y_edges[j]:.{title_ndigits}f})'
                    )
                    subsample_each.append(subsample & (bin_index[0] == i)
                                          & (bin_index[1] == j))

            # TODO: auto reshape according to bins
            if list_shape is not None:
                title_each = list_reshape(title_each, list_shape)
                subsample_each = list_reshape(subsample_each, list_shape)

            edges = [x_edges, y_edges]
        else:
            raise ValueError('can not handle more than two names')

        if return_edges:
            return subsample_each, title_each, edges
        return subsample_each, title_each

    def get_linear_combination_string(self,
                                      coefficients,
                                      names,
                                      string_format='.2f',
                                      with_unit=True) -> str:
        # TODO: constant term
        lc_str = f'{coefficients[0]:{string_format}} {self.get_label_by_name(names[0], with_unit=with_unit)}'
        for this_c, this_n in zip(coefficients[1:], names[1:]):
            sign = '+' if this_c > 0 else '-' if this_c < 0 else ''
            lc_str += f' {sign} {np.abs(this_c):{string_format}} {self.get_label_by_name(this_n)}'
        return lc_str

    def get_func_combination_string(self,
                                    names,
                                    func_name,
                                    with_unit=True) -> str:
        return f'{func_name}({", ".join([self.get_label_by_name(name, with_unit=with_unit) for name in names])})'
