import numpy as np
import pytest
from scipy.spatial.distance import cdist

from aa_utilities.computation import pairwise_distances


@pytest.fixture
def sample_matrices():
    rng = np.random.default_rng(seed=42)
    X = rng.standard_normal((50, 10))
    Y = rng.standard_normal((30, 10))
    return X, Y


# --- shape correctness ---

def test_all_vs_all_shape(sample_matrices):
    X, _ = sample_matrices
    D = pairwise_distances(X)
    assert D.shape == (50, 50)


def test_x_vs_y_shape(sample_matrices):
    X, Y = sample_matrices
    D = pairwise_distances(X, Y)
    assert D.shape == (50, 30)


# --- numerical correctness against scipy reference ---

def test_all_vs_all_values_match_scipy(sample_matrices):
    X, _ = sample_matrices
    D_expected = cdist(X, X, metric="euclidean")
    D = pairwise_distances(X, metric="euclidean", n_jobs=1)
    np.testing.assert_allclose(D, D_expected, rtol=1e-10)


def test_x_vs_y_values_match_scipy(sample_matrices):
    X, Y = sample_matrices
    D_expected = cdist(X, Y, metric="cosine")
    D = pairwise_distances(X, Y, metric="cosine", n_jobs=1)
    np.testing.assert_allclose(D, D_expected, rtol=1e-10)


def test_multicore_matches_single_core(sample_matrices):
    X, _ = sample_matrices
    D_single = pairwise_distances(X, metric="euclidean", n_jobs=1)
    D_multi = pairwise_distances(X, metric="euclidean", n_jobs=2)
    np.testing.assert_allclose(D_multi, D_single, rtol=1e-10)


# --- force_symmetry ---

def test_force_symmetry_produces_symmetric_matrix(sample_matrices):
    X, _ = sample_matrices
    D = pairwise_distances(X, metric="correlation", force_symmetry=True)
    np.testing.assert_allclose(D, D.T, atol=1e-12)


def test_force_symmetry_ignored_for_xy(sample_matrices):
    """force_symmetry should have no effect when Y is provided (non-square)."""
    X, Y = sample_matrices
    D_with = pairwise_distances(X, Y, force_symmetry=True)
    D_without = pairwise_distances(X, Y, force_symmetry=False)
    np.testing.assert_array_equal(D_with, D_without)


# --- callable metric ---

def test_custom_callable_metric(sample_matrices):
    X, _ = sample_matrices
    manhattan = lambda u, v: np.sum(np.abs(u - v))
    D_custom = pairwise_distances(X, metric=manhattan, n_jobs=1)
    D_expected = cdist(X, X, metric="cityblock")
    np.testing.assert_allclose(D_custom, D_expected, rtol=1e-10)


def test_custom_callable_selects_loky_backend(sample_matrices, monkeypatch):
    """When metric is callable, the auto-selected backend should be loky."""
    X, _ = sample_matrices
    captured = {}

    import aa_utilities.computation._distance as dist_module
    original_parallel = dist_module.Parallel

    class CapturingParallel(original_parallel):
        def __init__(self, **kwargs):
            captured["backend"] = kwargs.get("backend")
            super().__init__(**kwargs)

    monkeypatch.setattr(dist_module, "Parallel", CapturingParallel)
    pairwise_distances(X, metric=lambda u, v: float(np.sum((u - v) ** 2)), n_jobs=2)
    assert captured["backend"] == "loky"


def test_string_metric_selects_threading_backend(sample_matrices, monkeypatch):
    """When metric is a known GIL-free string, the auto-selected backend should be threading."""
    X, _ = sample_matrices
    captured = {}

    import aa_utilities.computation._distance as dist_module
    original_parallel = dist_module.Parallel

    class CapturingParallel(original_parallel):
        def __init__(self, **kwargs):
            captured["backend"] = kwargs.get("backend")
            super().__init__(**kwargs)

    monkeypatch.setattr(dist_module, "Parallel", CapturingParallel)
    pairwise_distances(X, metric="euclidean", n_jobs=2)
    assert captured["backend"] == "threading"


# --- input validation ---

def test_raises_on_1d_X():
    with pytest.raises(ValueError, match="'X' must be 2-D"):
        pairwise_distances(np.ones(10))


def test_raises_on_feature_mismatch():
    X = np.ones((10, 5))
    Y = np.ones((8, 7))
    with pytest.raises(ValueError, match="same number of features"):
        pairwise_distances(X, Y)


def test_raises_on_zero_chunk_size():
    X = np.ones((10, 5))
    with pytest.raises(ValueError, match="chunk_size"):
        pairwise_distances(X, chunk_size=0)


def test_n_chunks_capped_to_n_samples():
    """n_jobs > n_samples must not produce empty chunks or raise."""
    X = np.ones((3, 5))
    D = pairwise_distances(X, metric="euclidean", n_jobs=8)
    assert D.shape == (3, 3)


# --- metric kwargs forwarding ---

def test_metric_kwargs_forwarded(sample_matrices):
    X, _ = sample_matrices
    D_p3 = pairwise_distances(X, metric="minkowski", p=3, n_jobs=1)
    D_expected = cdist(X, X, metric="minkowski", p=3)
    np.testing.assert_allclose(D_p3, D_expected, rtol=1e-10)
