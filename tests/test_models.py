"""Tests for statistics functions within the Model layer."""

import pytest
import numpy as np
import numpy.testing as npt

from inflammation.models import daily_mean, daily_max, daily_min


@pytest.mark.parametrize(
    "test_input, test_result",
    [
        ([[0, 0], [0, 0], [0, 0]], [0, 0]),
        ([[1, 2], [3, 4], [5, 6]], [3, 4]),
        (np.zeros((3, 5)), np.zeros(5)),
    ],
)
def test_daily_mean(test_input, test_result):
    """Test that mean function works for both zeroes and integers."""

    npt.assert_array_equal(daily_mean(test_input), test_result)


def test_daily_mean_strings():
    """Test for TypeError when parsing strings."""

    with pytest.raises(TypeError):
        error_expected = daily_mean(["mean", "test"])  # noqa: F841


@pytest.mark.parametrize(
    "input_test",
    (["Helo", "There"]),
)
def test_daily_max_string(input_test):
    """Test for TypeError when parsing strings."""

    with pytest.raises(TypeError):
        error_expected = daily_max(input_test)  # noqa: F841


def test_daily_max_empty_array():
    """Test that daily_max raises ValueError when given an empty array."""
    with pytest.raises(ValueError):
        daily_max([])


@pytest.mark.parametrize(
    "test_input, test_result",
    [
        (np.array([[-1, 2], [3, -4], [5, 6]]), np.array([5, 6])),
        (np.array([[-1, 2], [3, -4], [-5, -6]]), np.array([3, 2])),
    ],
)
def test_daily_max(test_input, test_result):
    """Test that max function works for an array of positive and negative integers."""

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_max(test_input), test_result)


def test_daily_max_nan_propagation():
    """Test that max function selects NaN values as the maximum value for an array containing NaN's."""

    data = np.array([[1, np.nan], [3, 4]])
    result = daily_max(data)
    assert np.isnan(result[1])  # documents current behavior


@pytest.mark.parametrize(
    "test_input, test_result",
    [
        ([[0, 0, 0], [0, 0, 0], [0, 0, 0]], [0, 0, 0]),
        ([[1, 2, -1], [3, -2, 4], [5, -9, 6]], [1, -9, -1]),
        ([[0, 1, 2], [0, 3, 4]], [0, 1, 2]),
        ([[3, 3, 3], [3, 3, 3], [3, 3, 3]], [3, 3, 3]),
    ],
)
def test_daily_min(test_input, test_result):
    """Test that min function works for an array of positive and negative integers."""

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_min(test_input), test_result)
