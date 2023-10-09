import pytest
import numpy as np
from asa.utils import list_reshape, is_float, is_int, is_bool, balance_class


class TestUtils:
    def test_list_reshape(self):
        lst = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        shape = (3, 3)
        expected_output = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        assert list_reshape(lst, shape) == expected_output

        lst = [1, 2, 3, 4, 5, 6, 7, 8]
        shape = (2, 4)
        expected_output = [[1, 2, 3, 4], [5, 6, 7, 8]]
        assert list_reshape(lst, shape) == expected_output

    def test_is_float(self):

        assert is_float(0.1)
        assert is_float(np.float32(0.1))
        assert is_float(np.float64(0.1))
        assert is_float(np.array([0.1, 0.2, 0.3], dtype=np.float32))
        assert is_float(np.array([0.1, 0.2, 0.3], dtype=np.float64))
        assert is_float([0.1, 0.2, 0.3])

        assert not is_float(1)
        assert not is_float(np.int32(1))
        assert not is_float(np.int64(1))
        assert not is_float(np.array([1, 2, 3], dtype=np.int32))
        assert not is_float(np.array([1, 2, 3], dtype=np.int64))

        assert not is_float('a')
        assert not is_float(np.array(['a', 'b', 'c']))

        assert not is_float([1, 2, 3])
        assert not is_float([0.1, 0.2, 'a'])

        assert not is_float(np.ones(10, dtype=bool))
        assert not is_float(np.zeros(10, dtype=str))

    def test_is_int(self):

        assert not is_int(0.1)
        assert not is_int(np.float32(0.1))
        assert not is_int(np.float64(0.1))
        assert not is_int(np.array([0.1, 0.2, 0.3], dtype=np.float32))
        assert not is_int(np.array([0.1, 0.2, 0.3], dtype=np.float64))
        assert not is_int([0.1, 0.2, 0.3])

        assert is_int(1)
        assert is_int(np.int32(1))
        assert is_int(np.int64(1))
        assert is_int(np.array([1, 2, 3], dtype=np.int32))
        assert is_int(np.array([1, 2, 3], dtype=np.int64))
        assert is_int([1, 2, 3])

        assert not is_int('a')
        assert not is_int(np.array(['a', 'b', 'c']))
        assert not is_int([0.1, 0.2, 'a'])
        assert not is_int(np.ones(10, dtype=bool))
        assert not is_int(np.zeros(10, dtype=str))

    def test_is_bool(self):

        assert not is_bool(0.1)
        assert not is_bool(np.float32(0.1))
        assert not is_bool(np.float64(0.1))
        assert not is_bool(np.array([0.1, 0.2, 0.3], dtype=np.float32))
        assert not is_bool(np.array([0.1, 0.2, 0.3], dtype=np.float64))
        assert not is_bool([0.1, 0.2, 0.3])

        assert not is_bool(1)
        assert not is_bool(np.int32(1))
        assert not is_bool(np.int64(1))
        assert not is_bool(np.array([1, 2, 3], dtype=np.int32))
        assert not is_bool(np.array([1, 2, 3], dtype=np.int64))
        assert not is_bool([1, 2, 3])

        assert not is_bool('a')
        assert not is_bool(np.array(['a', 'b', 'c']))
        assert not is_bool([0.1, 0.2, 'a'])

        assert is_bool(True)
        assert is_bool(np.ones(10, dtype=bool))
        assert is_bool(np.zeros(10, dtype=bool))
        assert is_bool([True, False, True])

    def test_balance_class(self):

        x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        y = np.array([0, 1, 2])

        x_balanced, y_balanced = balance_class(x, y, random_state=0)

        assert x_balanced.shape == (3, 3)
        assert y_balanced.shape == (3, )

        x = np.random.normal(size=(100, 10))
        y = np.random.randint(0, 4, size=(100, ))
        y[y == 3] = 2

        x_balanced, y_balanced = balance_class(x, y, random_state=0)

        _, counts = np.unique(y, return_counts=True)

        assert x_balanced.shape == (3*counts.min(), 10)
        assert y_balanced.shape == (3*counts.min(), )

        for i in range(3):
            assert (y_balanced == i).sum() == (y_balanced == 0).sum()