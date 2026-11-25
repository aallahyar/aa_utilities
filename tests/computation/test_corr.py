import numpy as np
import pytest
from scipy import stats

from aa_utilities.computation._corr import spearman


def test_spearman_matches_scipy_with_ties():
    A = np.array([[1.0, 1.0, 2.0], [1.0, 2.0, 2.0], [3.0, 1.0, 2.0]])
    expected = np.array([[stats.spearmanr(a, b).statistic for b in A] for a in A])
    np.testing.assert_allclose(spearman(A), expected, atol=1e-12)


def test_spearman_x_vs_y_shape():
    rng = np.random.default_rng(0)
    assert spearman(rng.normal(size=(4, 10)), rng.normal(size=(3, 10))).shape == (4, 3)


def test_spearman_rejects_nan():
    with pytest.raises(ValueError):
        spearman(np.array([[1.0, np.nan, 3.0]]))
