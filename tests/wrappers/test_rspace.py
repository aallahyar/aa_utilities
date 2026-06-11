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
    reason='RSpace or its runtime dependencies are not available',
)

# ----- Initializations -----


@pytest.fixture
def rspace():
    if not RSPACE_AVAILABLE:
        pytest.skip('RSpace unavailable')
    return RSpace()


@pytest.fixture
def sample_df():
    rng = np.random.default_rng(seed=42)
    return pd.DataFrame(
        {
            'id': np.arange(1, 6, dtype=int),
            'grp': ['A', 'B', 'A', 'B', 'A'],
            'x': rng.integers(0, 100, size=5),
            'y': rng.normal(0.0, 1.0, size=5),
        }
    )


# ----- Scalars -----


@requires_rspace
def test_int_roundtrip(rspace):
    rspace['var_int'] = 3
    assert isinstance(rspace['var_int'], (int, np.integer))
    assert int(rspace['var_int']) == 3


@requires_rspace
def test_float_roundtrip(rspace):
    rspace['var_flt'] = 2.0
    out = rspace['var_flt']
    assert isinstance(out, (float, np.floating))
    assert out == pytest.approx(2.0)


@requires_rspace
def test_string_roundtrip(rspace):
    rspace['var_str'] = 'Hello, RSpace!'
    out = rspace['var_str']
    assert isinstance(out, str)
    assert out == 'Hello, RSpace!'


# ----- Sequence types -----


@requires_rspace
def test_list_roundtrip(rspace):
    py_list = ['Hello, RSpace!', 'This is a test.', 'asdfasd']
    rspace['lst'] = py_list
    out_list = rspace['lst']
    # Expect list;
    assert isinstance(out_list, (list, pd.Series))
    assert list(out_list) == py_list


@requires_rspace
def test_dict_roundtrip(rspace):
    py_dict = {'key1': 'val1', 'key2': 'val2', 'key3': 'val3'}
    rspace['dct'] = py_dict
    out_dict = rspace['dct']

    # Prefer exact dict round-trip
    if isinstance(out_dict, dict):
        assert out_dict == py_dict
    elif isinstance(out_dict, pd.Series):
        assert out_dict.to_dict() == py_dict
    # Fall back: explicitly fail on unexpected types
    else:
        pytest.fail(f'Unexpected dict round-trip type: {type(out_dict)}')


# ----- Numpy arrays -----


@requires_rspace
def test_vector_roundtrip(rspace):
    vec = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    rspace['vec'] = vec
    out_vec = rspace['vec']

    # Accept array-like; coerce for comparison
    out_vec_np = np.array(out_vec, dtype=float).reshape(-1)
    assert out_vec_np.shape == vec.shape
    assert np.allclose(out_vec_np, vec)


@requires_rspace
def test_matrix_roundtrip(rspace):
    mat = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    rspace['arr_2d'] = mat
    out_mat = rspace['arr_2d']

    out_mat_np = np.array(out_mat, dtype=float)
    assert out_mat_np.shape == mat.shape
    assert np.allclose(out_mat_np, mat)


# ----- Pandas DataFrame -----


@requires_rspace
def test_dataframe_roundtrip(rspace, sample_df):
    rspace['df'] = sample_df
    out_df = rspace['df']

    # Coerce to pandas if needed
    if not isinstance(out_df, pd.DataFrame):
        out_df = pd.DataFrame(out_df)

    # Shape and columns set must match (order can differ)
    assert out_df.shape == sample_df.shape
    assert list(out_df.columns) == list(sample_df.columns)

    # Value checks with dtype tolerance
    for col in sample_df.columns:
        if pd.api.types.is_numeric_dtype(sample_df[col]):
            left = pd.to_numeric(out_df[col], errors='coerce').to_numpy(dtype=float)
            right = sample_df[col].to_numpy(dtype=float)
            assert np.allclose(left, right, equal_nan=True, atol=1e-8, rtol=1e-8)
        else:
            assert list(out_df[col].astype(str)) == list(sample_df[col].astype(str))


# ----- __call__ behavior -----


@requires_rspace
def test_call_scalar_conversion(rspace):
    out = rspace('1 + 2')
    assert isinstance(out, (int, float, np.integer, np.floating))
    assert float(out) == pytest.approx(3.0)


@requires_rspace
def test_call_null_conversion(rspace):
    out = rspace('NULL')
    assert out is None


@requires_rspace
def test_call_without_conversion_returns_r_object(rspace):
    out = rspace('1 + 2', convert=False)
    assert not isinstance(out, (int, float, np.integer, np.floating))


@requires_rspace
def test_getitem_missing_key_raises_keyerror(rspace):
    with pytest.raises(KeyError):
        _ = rspace['__definitely_missing_object__']


# ----- Helpers for nested-structure tests -----


def _assert_df_values_equal(actual: pd.DataFrame, expected: pd.DataFrame) -> None:
    """Value-level equality check with dtype tolerance (mirrors test_dataframe_roundtrip logic)."""
    assert actual.shape == expected.shape
    assert list(actual.columns) == list(expected.columns)
    for col in expected.columns:
        if pd.api.types.is_numeric_dtype(expected[col]):
            left = pd.to_numeric(actual[col], errors='coerce').to_numpy(dtype=float)
            right = expected[col].to_numpy(dtype=float)
            assert np.allclose(left, right, equal_nan=True, atol=1e-8, rtol=1e-8)
        else:
            assert list(actual[col].astype(str)) == list(expected[col].astype(str))


@pytest.fixture
def two_dataframes():
    """Two simple numeric DataFrames for nested-structure tests."""
    rng = np.random.default_rng(seed=0)
    df1 = pd.DataFrame({'a': rng.integers(0, 10, 4).astype(float), 'b': rng.normal(size=4)})
    df2 = pd.DataFrame({'x': rng.integers(0, 10, 3).astype(float), 'y': rng.normal(size=3)})
    return df1, df2


# ----- R → Python: generic list types -----


@requires_rspace
def test_named_list_of_dataframes_from_r(rspace):
    rspace('''
    df1 <- data.frame(a = c(1.0, 2.0, 3.0), b = c(4.0, 5.0, 6.0))
    df2 <- data.frame(x = c(7.0, 8.0), y = c(9.0, 10.0))
    lst <- list(first = df1, second = df2)
    ''')
    out = rspace['lst']
    assert isinstance(out, dict)
    assert set(out.keys()) == {'first', 'second'}
    assert isinstance(out['first'], pd.DataFrame)
    assert isinstance(out['second'], pd.DataFrame)
    assert list(out['first'].columns) == ['a', 'b']
    assert list(out['second'].columns) == ['x', 'y']


@requires_rspace
def test_unnamed_list_of_dataframes_from_r(rspace):
    rspace('''
    df1 <- data.frame(a = c(1.0, 2.0))
    df2 <- data.frame(b = c(3.0, 4.0))
    lst <- list(df1, df2)
    ''')
    out = rspace['lst']
    assert isinstance(out, list)
    assert len(out) == 2
    assert all(isinstance(el, pd.DataFrame) for el in out)


@requires_rspace
def test_nested_named_list_from_r(rspace):
    rspace('''
    df <- data.frame(v = c(1.0, 2.0, 3.0))
    nested <- list(outer = list(inner = df))
    ''')
    out = rspace['nested']
    assert isinstance(out, dict)
    assert isinstance(out['outer'], dict)
    assert isinstance(out['outer']['inner'], pd.DataFrame)
    assert list(out['outer']['inner'].columns) == ['v']


@requires_rspace
def test_mixed_type_named_list_from_r(rspace):
    rspace("mixed <- list(a = 42L, b = 'hello', c = c(1.0, 2.0, 3.0))")
    out = rspace['mixed']
    assert isinstance(out, dict)
    assert set(out.keys()) == {'a', 'b', 'c'}
    assert int(out['a']) == 42
    assert str(out['b']) == 'hello'
    assert isinstance(out['c'], pd.Series)
    assert np.allclose(out['c'].to_numpy(dtype=float), [1.0, 2.0, 3.0])


@requires_rspace
def test_null_inside_named_list_from_r(rspace):
    rspace("lst_null <- list(a = NULL, b = 42L)")
    out = rspace['lst_null']
    assert isinstance(out, dict)
    assert out['a'] is None
    assert int(out['b']) == 42


@requires_rspace
def test_length1_named_list_is_dict_not_scalar(rspace):
    """A named list of length 1 must return a dict, not be coerced to a scalar."""
    rspace("single <- list(a = 42L)")
    out = rspace['single']
    assert isinstance(out, dict)
    assert int(out['a']) == 42


@requires_rspace
def test_dataframe_metadata_preserved_inside_list(rspace):
    """Column names and row names of a DataFrame must survive extraction from a named list."""
    rspace('''
    df <- data.frame(col_a = c(1.0, 2.0, 3.0), col_b = c(4.0, 5.0, 6.0))
    rownames(df) <- c('r1', 'r2', 'r3')
    wrapped <- list(data = df)
    ''')
    out = rspace['wrapped']
    df_out = out['data']
    assert isinstance(df_out, pd.DataFrame)
    assert list(df_out.columns) == ['col_a', 'col_b']
    assert list(df_out.index) == ['r1', 'r2', 'r3']


@requires_rspace
def test_duplicate_names_raises_valueerror(rspace):
    rspace("dupes <- list(a = 1, a = 2)")
    with pytest.raises(ValueError, match='duplicate'):
        _ = rspace['dupes']


# ----- Python → R → Python: round-trips for nested types -----


@requires_rspace
def test_dict_of_dataframes_roundtrip(rspace, two_dataframes):
    df1, df2 = two_dataframes
    rspace['dfs'] = {'first': df1, 'second': df2}
    out = rspace['dfs']
    assert isinstance(out, dict)
    assert set(out.keys()) == {'first', 'second'}
    _assert_df_values_equal(out['first'], df1)
    _assert_df_values_equal(out['second'], df2)


@requires_rspace
def test_nested_dict_roundtrip(rspace, two_dataframes):
    df1, _ = two_dataframes
    rspace['nested'] = {'outer': {'inner': df1}}
    out = rspace['nested']
    assert isinstance(out, dict)
    assert isinstance(out['outer'], dict)
    _assert_df_values_equal(out['outer']['inner'], df1)


@requires_rspace
def test_list_of_dataframes_roundtrip(rspace, two_dataframes):
    df1, df2 = two_dataframes
    rspace['lst'] = [df1, df2]
    out = rspace['lst']
    assert isinstance(out, list)
    assert len(out) == 2
    _assert_df_values_equal(out[0], df1)
    _assert_df_values_equal(out[1], df2)
