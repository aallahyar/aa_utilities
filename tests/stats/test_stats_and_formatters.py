import numpy as np
import pytest

from aa_utilities.helpers.formatters import pvalue_to_asterisks
from aa_utilities.stats import IterativeStats


def test_empty_iterative_stats_are_all_nan():
    stats = IterativeStats()
    assert np.isnan(stats.mean()) and np.isnan(stats.var()) and np.isnan(stats.std())


def test_iterative_stats_match_numpy():
    data = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
    stats = IterativeStats()
    for value in data:
        stats.include(value)
    assert stats.mean() == pytest.approx(data.mean())
    assert stats.var() == pytest.approx(data.var())


@pytest.mark.parametrize('missing', [np.nan, None])
def test_missing_pvalue_is_not_reported_as_not_significant(missing):
    assert pvalue_to_asterisks(missing) == 'NA'


@pytest.mark.parametrize('p_value, expected', [(0.00001, '****'), (0.0005, '***'), (0.005, '**'), (0.03, '*'), (0.2, 'ns')])
def test_pvalue_to_asterisks_thresholds(p_value, expected):
    assert pvalue_to_asterisks(p_value) == expected
