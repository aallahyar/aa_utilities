import numpy as np
import pandas as pd
import pytest

from aa_utilities.helpers.pandas import describe, reorder_by_similarity, search, select, sort


def test_sort_series_with_duplicated_index_keeps_every_row_once():
    srs = pd.Series(['b', 'a', 'b'], index=[0, 0, 1])
    out = sort(srs, ['a', 'b'])
    assert out.tolist() == ['a', 'b', 'b']
    assert len(out) == 3


def test_sort_dataframe_with_duplicated_index_keeps_every_row_once():
    df = pd.DataFrame({'k': ['b', 'a'], 'v': [1, 2]}, index=[5, 5])
    out = sort(df, {'k': ['a', 'b']})
    assert out['v'].tolist() == [2, 1]


def test_search_reports_original_column_positions():
    df = pd.DataFrame({'text': ['a'], 'number': [7]})
    result = search(df, 7)
    assert result['col_name'].tolist() == ['number']
    assert result['col_number'].tolist() == [1]


def test_search_handles_nullable_missing_values():
    df = pd.DataFrame({'x': pd.Series([1, pd.NA], dtype='Int64'), 's': pd.Series(['a', pd.NA], dtype='string')})
    assert search(df, 1)['row_number'].tolist() == [0]
    assert search(df, 'a')['row_number'].tolist() == [0]


@pytest.mark.parametrize('value, kwargs', [('abc', {'case': False}), ('b', {'regex': True})])
def test_search_string_ignores_columns_holding_only_numbers(value, kwargs):
    df = pd.DataFrame({'num': pd.Series([1, 2], dtype=object), 'txt': ['Abc', 'x']})
    result = search(df, value, **kwargs)
    assert result['col_name'].tolist() == ['txt']


def test_select_without_queries_raises():
    df = pd.DataFrame({'a': [1]})
    with pytest.raises(ValueError):
        select(df, [])
    with pytest.raises(ValueError):
        select(df, {})


def test_describe_returns_tables_with_default_flags():
    df = pd.DataFrame({'n': [1.0, 2.0], 'c': ['a', 'b'], 's': pd.Series(['x', 'y'], dtype='string')})
    numeric, categorical = describe(df, return_dfs=True)
    assert numeric.index.tolist() == ['n']
    assert categorical.index.tolist() == ['c', 's']


def test_describe_returns_tables_when_display_flags_are_off():
    df = pd.DataFrame({'n': [1.0, 2.0], 'c': ['a', 'b']})
    numeric, categorical = describe(df, show_numeric=False, show_categorical=False, return_dfs=True)
    assert numeric is not None and categorical is not None


def test_reorder_by_similarity_accepts_nullable_integers():
    df = pd.DataFrame({'a': pd.Series([1, 5, 2], dtype='Int64'), 'b': pd.Series([1, 6, 2], dtype='Int64')})
    out = reorder_by_similarity(df)
    assert out.shape == df.shape
    assert sorted(out.index) == [0, 1, 2]


def test_reorder_by_similarity_rejects_missing_values():
    with pytest.raises(ValueError):
        reorder_by_similarity(pd.DataFrame({'a': [1.0, np.nan], 'b': [1.0, 2.0]}))
