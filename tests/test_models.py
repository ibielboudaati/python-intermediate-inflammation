"""Tests for statistics functions within the Model layer."""
import numpy.testing as npt
import numpy as np
import pytest
from inflammation.models import daily_mean, daily_max,daily_min


@pytest.mark.parametrize(
    "test, expected",
    [
        ([ [0, 0], [0, 0], [0, 0] ], [0, 0]),
        ([ [1, 2], [3, 4], [5, 6] ], [3, 4]),
    ])
def test_daily_mean(test, expected):
    """Test mean function works for array of zeroes and positive integers."""
    npt.assert_array_equal(daily_mean(np.array(test)), np.array(expected))

@pytest.mark.parametrize(
    "test, expected",
      [
        ([ [2, 5], [1, 8], [5, 7] ], [5, 8]),
        ([ [-1, -2], [-3, -4], [-5, -6] ], [-1, -2]),
        ([[2,-5], [-3,3],[6,8]],[6,8]),
    ])
def test_daily_max(test, expected):
    """Test function checking the working of maximum function for different integer situations"""
    npt.assert_array_equal(daily_max(np.array(test)), np.array(expected))

@pytest.mark.parametrize(
    "test, expected",
      [
        ([ [2, 5], [1, 8], [5, 7] ], [1, 5]),
        ([ [-1, -2], [-3, -4], [-5, -6] ], [-5, -6]),
        ([[2,-5], [-3,3],[6,8]], [-3,-5]),
    ])
def test_daily_min(test, expected):
    """Test function checking the working of minimum function for different integer situations"""
    npt.assert_array_equal(daily_min(np.array(test)), np.array(expected))
