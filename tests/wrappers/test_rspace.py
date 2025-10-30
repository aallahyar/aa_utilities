# tests/wrappers/test_rspace.py

import pytest
import numpy as np
import pandas as pd

try:
    from aa_utilities.wrappers import RSpace
    RSPACE_AVAILABLE = True
except Exception:
    RSPACE_AVAILABLE = False

requires_rspace = pytest.mark.skipif(
    not RSPACE_AVAILABLE,
    reason="RSpace or its runtime dependencies are not available",
)

# ----- Initializations -----

@pytest.fixture
def rspace():
    if not RSPACE_AVAILABLE:
        pytest.skip("RSpace unavailable")
    return RSpace()

@pytest.fixture
def sample_df():
    rng = np.random.default_rng(seed=42)
    return pd.DataFrame(
        {
            "id": np.arange(1, 6, dtype=int),
            "grp": ["A", "B", "A", "B", "A"],
            "x": rng.integers(0, 100, size=5),
            "y": rng.normal(0.0, 1.0, size=5),
        }
    )

# ----- Scalars -----

@requires_rspace
def test_int_roundtrip(rspace):
    rspace["var_int"] = 3
    assert isinstance(rspace["var_int"], (int, np.integer))
    assert int(rspace["var_int"]) == 3

@requires_rspace
def test_float_roundtrip(rspace):
    rspace["var_flt"] = 2.0
    out = rspace["var_flt"]
    assert isinstance(out, (float, np.floating))
    assert out == pytest.approx(2.0)

@requires_rspace
def test_string_roundtrip(rspace):
    rspace["var_str"] = "Hello, RSpace!"
    out = rspace["var_str"]
    assert isinstance(out, str)
    assert out == "Hello, RSpace!"

# ----- Sequence types -----

@requires_rspace
def test_list_roundtrip(rspace):
    py_list = ["Hello, RSpace!", "This is a test.", "asdfasd"]
    rspace["lst"] = py_list
    out_list = rspace["lst"]
    # Expect list;
    assert isinstance(out_list, (list, pd.Series))
    assert list(out_list) == py_list

@requires_rspace
def test_dict_roundtrip(rspace):
    py_dict = {"key1": "val1", "key2": "val2", "key3": "val3"}
    rspace["dct"] = py_dict
    out_dict = rspace["dct"]

    # Prefer exact dict round-trip
    if isinstance(out_dict, dict):
        assert out_dict == py_dict
    elif isinstance(out_dict, pd.Series):
        assert out_dict.to_dict() == py_dict
    # Fall back: explicitly fail on unexpected types
    else:
        pytest.fail(f"Unexpected dict round-trip type: {type(out_dict)}")

# ----- Numpy arrays -----

@requires_rspace
def test_vector_roundtrip(rspace):
    vec = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    rspace["vec"] = vec
    out_vec = rspace["vec"]

    # Accept array-like; coerce for comparison
    out_vec_np = np.array(out_vec, dtype=float).reshape(-1)
    assert out_vec_np.shape == vec.shape
    assert np.allclose(out_vec_np, vec)

@requires_rspace
def test_matrix_roundtrip(rspace):
    mat = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    rspace["arr_2d"] = mat
    out_mat = rspace["arr_2d"]

    out_mat_np = np.array(out_mat, dtype=float)
    assert out_mat_np.shape == mat.shape
    assert np.allclose(out_mat_np, mat)

# ----- Pandas DataFrame -----

@requires_rspace
def test_dataframe_roundtrip(rspace, sample_df):
    rspace["df"] = sample_df
    out_df = rspace["df"]

    # Coerce to pandas if needed
    if not isinstance(out_df, pd.DataFrame):
        out_df = pd.DataFrame(out_df)

    # Shape and columns set must match (order can differ)
    assert out_df.shape == sample_df.shape
    assert list(out_df.columns) == list(sample_df.columns)

    # Value checks with dtype tolerance
    for col in sample_df.columns:
        if pd.api.types.is_numeric_dtype(sample_df[col]):
            left = pd.to_numeric(out_df[col], errors="coerce").to_numpy(dtype=float)
            right = sample_df[col].to_numpy(dtype=float)
            assert np.allclose(left, right, equal_nan=True, atol=1e-8, rtol=1e-8)
        else:
            assert list(out_df[col].astype(str)) == list(sample_df[col].astype(str))
