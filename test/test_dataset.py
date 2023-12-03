import pytest
import numpy as np
from asa.dataset import Dataset
from asa.dataset import parse_inequality
import asa.uncertainty as unc


class TestDataset:

    def gen_dataset(self):
        x = np.arange(10)
        y = np.arange(10) * 2
        z = np.arange(10) * 3
        dataset = Dataset(
            np.array([x, y, z]).T, ['x', 'y', 'z'],
            ['x label', 'y label', 'z label'])
        return dataset, x, y, z

    def test_getitem(self):
        dataset, x, y, z = self.gen_dataset()

        assert np.array_equal(dataset[0], np.array([x[0], y[0], z[0]]))
        assert np.array_equal(dataset[:, :], np.array([x, y, z]).T)
        assert np.array_equal(dataset[:, 0], x)
        assert np.array_equal(dataset[:, ['x', 'y']], np.array([x, y]).T)
        assert np.array_equal(dataset[:, 'x'], x)
        assert np.array_equal(dataset[['x', 'y']], np.array([x, y]).T)

        with pytest.raises(ValueError) as excinfo:
            dataset[0, 0, 0]
        assert 'key should be a tuple of length 2' == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            dataset['x', 0]
        assert 'key[0] can not be string or list of string' == str(
            excinfo.value)

    def test_parse_inequality(self):

        assert (parse_inequality("1<=2<=x") == ['1', '<=', '2', '<=', 'x'])
        assert (parse_inequality("1>2>=x") == ['1', '>', '2', '>=', 'x'])
        assert (parse_inequality("1 <= 2 <= x") == ['1', '<=', '2', '<=', 'x'])
        assert parse_inequality("10<20>30>=40<=50") == [
            '10', '<', '20', '>', '30', '>=', '40', '<=', '50'
        ]
        assert parse_inequality("5<=10") == ['5', '<=', '10']
        assert parse_inequality("15>=5") == ['15', '>=', '5']
        assert parse_inequality("100<200") == ['100', '<', '200']
        assert parse_inequality("30>20") == ['30', '>', '20']
        assert parse_inequality("x<100") == ['x', '<', '100']
        assert parse_inequality("a>=b") == ['a', '>=', 'b']

    def test_inequality_to_subsample(self):

        dataset, x, y, z = self.gen_dataset()

        debug = True

        assert np.array_equal(
            dataset.inequality_to_subsample("x<5", debug=debug),
            np.array(x < 5))
        assert np.array_equal(
            dataset.inequality_to_subsample("x<=5", debug=debug),
            np.array(x <= 5))
        assert np.array_equal(
            dataset.inequality_to_subsample("x>y", debug=debug),
            np.array(x > y))
        assert np.array_equal(
            dataset.inequality_to_subsample("x==y", debug=debug),
            np.array(x == y))
        assert np.array_equal(
            dataset.inequality_to_subsample("x==5", debug=debug),
            np.array(x == 5))

        dataset['t'] = x + 2
        assert np.array_equal(
            dataset.inequality_to_subsample("log10@t<np.log10(5)",
                                            debug=debug),
            np.array(np.log10(x + 2) < np.log10(5)))
        assert np.array_equal(
            dataset.inequality_to_subsample("square@t<3", debug=debug),
            np.array(np.square(x + 2) < 3))

        # operation
        assert np.array_equal(
            dataset.inequality_to_subsample("x<=3+2", debug=debug),
            np.array(x <= 5))
        assert np.array_equal(
            dataset.inequality_to_subsample("x/5<=1", debug=debug),
            np.array(x <= 5))
        assert np.array_equal(
            dataset.inequality_to_subsample("x + y<=5", debug=debug),
            np.array((x + y) <= 5))

        # and, or
        assert np.array_equal(
            dataset.inequality_to_subsample("x > 3 & y > 5", debug=debug),
            np.array((x > 3) & (y > 5)))
        assert np.array_equal(
            dataset.inequality_to_subsample("x > 3 | y > 5", debug=debug),
            np.array((x > 3) | (y > 5)))
        assert np.array_equal(
            dataset.inequality_to_subsample("x > 3 | y == 1", debug=debug),
            np.array((x > 3) | (y == 1)))

    def test_check_same_length(self):

        _, x, y, z = self.gen_dataset()

        # this error is handled by pd.DataFrame
        with pytest.raises(ValueError) as excinfo:
            Dataset(
                np.array([x, y, z]).T, ['x', 'y'],
                ['x label', 'y label', 'z label'])
        assert 'Shape of passed values is (10, 3), indices imply (10, 2)' == str(
            excinfo.value)

        with pytest.raises(IndexError) as excinfo:
            Dataset(
                np.array([x, y, z]).T, ['x', 'y', 'z'], ['x label', 'y label'])
        assert 'list index out of range' == str(excinfo.value)

        Dataset(
            np.array([x, y, z]).T, ['x', 'y', 'z'],
            ['x label', 'y label', 'z label'])

    def test_setitem(self):

        dataset, x, y, z = self.gen_dataset()
        dataset['x'] = x * 2
        assert np.array_equal(dataset['x'], x * 2)

        dataset, x, y, z = self.gen_dataset()
        dataset[:, 'x'] = x * 2
        assert np.array_equal(dataset['x'], x * 2)

        dataset, x, y, z = self.gen_dataset()
        dataset[:, ['x', 'y']] = np.array([x * 2, y * 3]).T
        assert np.array_equal(dataset['x'], x * 2)
        assert np.array_equal(dataset['y'], y * 3)

        dataset, x, y, z = self.gen_dataset()
        dataset['x2'] = x * 2
        assert np.array_equal(dataset['x2'], x * 2)

    def test_construct(self):
        _, x, y, z = self.gen_dataset()

        # label as dict
        dataset = Dataset(np.array([x, y, z]).T,
                          names=['x', 'y', 'z'],
                          labels={
                              'x': 'x_label',
                              'y': 'y_label',
                              'z': 'z_label'
                          })
        assert dataset.labels == {
            'x': 'x_label',
            'y': 'y_label',
            'z': 'z_label'
        }

        dataset = Dataset(np.array([x, y, z]).T,
                          names=['x', 'y', 'z'],
                          labels={
                              'x': 'x_label',
                              'y': 'y_label'
                          })
        assert dataset.labels == {'x': 'x_label', 'y': 'y_label'}

    def test_update(self):

        dataset, x, y, z = self.gen_dataset()
        assert dataset.labels == {
            'x': 'x label',
            'y': 'y label',
            'z': 'z label'
        }

        dataset.update_labels({'x': 'xxx', 'y': 'yyy'})
        assert dataset.labels == {'x': 'xxx', 'y': 'yyy', 'z': 'z label'}

        dataset.update_names({'x': 'x1'})
        assert np.array_equal(dataset.names, np.array(['x1', 'y', 'z']))
        assert np.array_equal(np.asarray(dataset.data.columns), dataset.names)

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")  # np.log10(0)
    def test_get_range_by_name(self):

        _, x, y, z = self.gen_dataset()
        dataset = Dataset(np.array([x, y, z]).T, ['x', 'y', 'z'],
                          ['x label', 'y label', 'z label'],
                          ranges={'x': [0, 9]})

        assert dataset.get_range_by_name('x') == [0, 9]
        assert dataset.get_range_by_name('log10@x') is None
        assert dataset.get_range_by_name('y') is None

        dataset = Dataset(np.array([x, y, z]).T, ['x', 'y', 'z'],
                          ['x label', 'y label', 'z label'],
                          ranges={'x': [1, 9]})

        assert dataset.get_range_by_name('log10@x') == [
            np.log10(1), np.log10(9)
        ]

    def test_get_data_by_name(self):

        dataset, x, y, z = self.gen_dataset()

        assert np.array_equal(dataset.get_data_by_name('x'), x)
        assert np.array_equal(dataset.get_data_by_names(['x', 'y']),
                              np.array([x, y]).T)

        x = np.arange(10) + 1
        x_err = (np.arange(10) + 1) * 0.1
        dataset = Dataset(np.array([x, x_err]).T, ['x', 'x_err'])
        assert np.array_equal(dataset.get_data_by_name('x'), x)
        assert np.array_equal(dataset.get_data_by_name('x_err'), x_err)
        assert np.allclose(dataset.get_data_by_name('x_snr'),
                           np.abs(x / x_err))

        assert np.allclose(dataset.get_data_by_name('log10@x'), np.log10(x))

        assert np.allclose(dataset.get_data_by_name('x_snr'),
                           np.abs(x / x_err))
        assert np.allclose(dataset.get_data_by_name('log10@x_err'),
                           unc.log10(x, x_err))
        assert np.allclose(dataset.get_data_by_name('log10@x_snr'),
                           np.abs(np.log10(x) / unc.log10(x, x_err)))

        dataset = Dataset(np.array([x, x_err]).T, ['x', 'x_errerr'],
                          err_postfix='errerr')
        assert np.array_equal(dataset.get_data_by_name('x_errerr'), x_err)

        with pytest.raises(ValueError) as excinfo:
            dataset.get_data_by_name('x_err')
        assert "'x_err' is not in list" == str(excinfo.value)

        x_snr = x / x_err
        dataset = Dataset(np.array([x, x_snr]).T, ['x', 'x_snr'])
        assert np.array_equal(dataset.get_data_by_name('x_snr'), x_snr)
        assert np.allclose(dataset.get_data_by_name('x_err'), x_err)
        assert np.allclose(dataset.get_data_by_name('log10@x_snr'),
                           unc.log10_snr(x, x_snr))
        assert np.allclose(dataset.get_data_by_name('log10@x_err'),
                           unc.log10(x, x_err))

        # potential false positive
        dataset = Dataset(
            np.array([x, x_snr, x, x]).T, ['x', 'x_snr', 'snr_x', 'xsnr'])
        assert np.array_equal(dataset.get_data_by_name('snr_x'), x)
        assert np.array_equal(dataset.get_data_by_name('xsnr'), x)

    def test_remove_postfix(self):
        dataset, x, y, z = self.gen_dataset()
        assert dataset.remove_snr_postfix('x_snr') == 'x'
        with pytest.raises(ValueError) as excinfo:
            dataset.remove_snr_postfix('x')
        assert "x does not end with _snr" == str(excinfo.value)

        assert dataset.remove_err_postfix('x_err') == 'x'
        with pytest.raises(ValueError) as excinfo:
            dataset.remove_err_postfix('x')
        assert "x does not end with _err" == str(excinfo.value)

    def test_is_legal_name(self):
        dataset, x, y, z = self.gen_dataset()
        assert dataset.is_legal_name('x')
        assert dataset.is_legal_name('y')
        assert dataset.is_legal_name('z')
        assert dataset.is_legal_name('x_err')
        assert dataset.is_legal_name('x_snr')
        assert dataset.is_legal_name('log10@x')
        assert dataset.is_legal_name('log10@x_err')

        assert not dataset.is_legal_name('t')
        assert not dataset.is_legal_name('t_err')
        assert not dataset.is_legal_name('t_snr')
        assert not dataset.is_legal_name('log10@t')
        assert not dataset.is_legal_name('log10@t_err')
