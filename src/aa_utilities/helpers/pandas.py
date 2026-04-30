from typing import (
    Any,
    Callable,
    Union,
    Literal,
)

import numpy as np
import pandas as pd


def generate_dataframe(n=100, seed=42):
    import datetime

    # initializations
    rng = np.random.default_rng(seed=seed)

    df = pd.DataFrame(
        {
            'A': rng.normal(loc=0, scale=1, size=n),
            'B': rng.uniform(low=0, high=1, size=n),
            'C': rng.integers(low=0, high=100, size=n),
            'D': rng.exponential(scale=1, size=n),
            'E': rng.choice(['flour', 'egg', 'oil', 'milk', 'water', 'salt', 'suger'], size=n),
            'F': rng.choice([f'str_{i}' for i in range(10)], size=n),
            'G': rng.choice(pd.date_range(datetime.datetime(2023, 1, 1), datetime.datetime(2024, 1, 1)), size=n),
        }
    )
    return df


def is_true(
    data: Any, condition: Union[Callable, bool], message: Union[Callable, str] = 'The `condition` argument is False!'
):
    """Returns the data, if `condition` is True
    This is useful as a convenience function during Pandas chainig operations and
    data manupulation to check if a condision is True.

    Args:
        data (any): The object to be returned, if the condition is True.
        condition (bool): The condition to be tested
        message (str, func, optional): Messsage to be shown
            if the condition is False. Defaults to 'Condition is False!'.

    Returns:
        any: The `data` as it is provided (i.e., no modification).

    Examples:
        print(
            pd.DataFrame({
                'a': range(5),
                'b': range(100, 105),
                'c': range(200, 205),
            })
            .pipe(is_true, condition=True)
            .pipe(is_true, True)
            .pipe(is_true, lambda df: df.b.gt(50).all())
            .pipe(lambda df: is_true(data=df, condition=True))
            .pipe(lambda df: is_true(df.iloc[:-2], condition=df.c.ge(200).all())) # note: df.iloc[:-2] is returned
            .pipe(lambda df: is_true(df.iloc[:-2], condition=df.c.ge(200).all()) and df) # note: df is returned
            .pipe(lambda df: is_true(df.iloc[1:], df.a.ge(0).all())) # note: df.iloc[1:] is returned
        )

        #    a    b    c
        # 1  1  101  201
        # 2  2  102  202

    """

    # perform the test
    if callable(condition):
        result = condition(data)
    else:
        result = condition

    # check the result
    if not result:
        if callable(message):
            message_str = message(data)
        else:
            message_str = message
        raise ValueError(message_str)

    return data


def select(dataframe: pd.DataFrame, queries: Union[str, dict, list], indicator='query'):
    """Selects subsets of rows from a given DataFrame according to a dictionary of queries
    The resulting rows from each query is indicated in `indicator` column.

    Example:
        import pandas as pd
        df = pd.DataFrame({
            'id': list('AABAB'),
            'day1': [23, 25, 27, 26, 24],
            'day2': [22, 21, 25, 26, 23],
        })

        (
            df
            .pipe(select, 'id in ["A"] and day1 >= 25') # this is equivalent to .query()
            .pipe(select, ['id in ["A"]', 'day1 >= 25']) # two queries, results are indicated by 0, 1
            .pipe(select, {'set1': 'id in ["A"]', 'set2': 'day1 >= 25'})
        )
    """

    # sanity checks
    assert isinstance(dataframe, pd.DataFrame), '`dataframe` must be a `pd.DataFrame` instance'
    assert isinstance(queries, (str, dict, list)), '`queries` must be either a `str`, `list`, or `dict`'
    assert isinstance(indicator, str), '`indicator` must be a `str`'
    assert indicator not in dataframe.columns, f'`indicator` column "{indicator}" already exists in the dataframe!'

    if isinstance(queries, str):
        queries = {'': queries}
    if isinstance(queries, list):
        queries = {idx: que for idx, que in enumerate(queries)}

    # select subsets of dataframe based on each query
    subsets = []
    for que_id, que_str in queries.items():
        subsets.append(
            dataframe.query(que_str)
            .copy(deep=True)
            .assign(
                **{
                    indicator: que_id,
                }
            )
        )

    merged = pd.concat(subsets, axis=0, ignore_index=False, sort=False, copy=True)

    return merged


def store(data: Any, namespace: dict, name: str = 'stored_data', copy=True) -> Any:
    """Stores the data into a provided (dict) namespace.
    This function is useful during Pandas chaining operations to store an
    intermediate dataframe in the middle of the chain.
    The original provided `data` is returned.

    Note: globals() is referring to the current modules namespace, and not the global in
        which the `main` function is operating: https://stackoverflow.com/a/60359384/1397843
        Therefore, the `namespace` argument can not have a default value of `globals()`.
        Otherwise, it will store the variable into the current module's (i.e., `convenience`) global
        name space.

    Args:
        data (any): The object to be stored.
        namespace (dict): The container objects in which the data will be stored in.
        name (str): Name of the variable that will hold the stored data.
        copy (bool): Whether or not copy the data, or store the reference to the data.

    Returns:
        any: The `data` as it is provided (i.e., no modification).

    Examples:
        import pandas as pd

        container = {}
        df = (
            pd.DataFrame({
                'i': list('ABCDE'),
                'a': range(5),
                'b': range(100, 105),
                'c': range(200, 205),
            })
            .pipe(store, namespace=globals()) # store the current `df` into global `stored_data` variable
            .pipe(store, namespace=globals(), name='test1')
            .pipe(store, namespace=container, name='test2')
            .pipe(store, namespace=container, name='test3', copy=False)

            # store a different variable, but still return the originally given `df`
            .pipe(lambda df: store({'test': 'value1'}, namespace=globals(), name='test4') and df)

            # modify the given `df`, store it, return the modified `df`
            .pipe(lambda df: store(df.iloc[:-2], namespace=globals(), name='test5'))
        )
        print('>df\n', df) # df with the last two rows removed
        df.iloc[0, 0] = 'X'
        print('>stored_data\n', stored_data)
        print('>container\n', container)
        print('>test1\n', test1)
        print('>test3\n', test3)
        print('test2' in globals())
        print('test3' in globals())
        print('>test4\n', test4)
        print('>test5\n', test5)
    """
    import copy as _copy

    if copy:
        namespace[name] = _copy.deepcopy(data)
    else:
        namespace[name] = data
    return data


def sort(
    data: Union[pd.Series, pd.DataFrame],
    orders: Union[list, tuple, np.ndarray, pd.Index, dict],
    ascending=True,
    method='mergesort',
    na_position: Literal['first', 'last'] = 'last',
    validate=True,
) -> Union[pd.DataFrame, pd.Series]:
    """Sorts the given data according to explicit, user-defined orders.

    The function uses categorical sorting so the provided order is respected.
    Values not listed in `orders` are treated as missing and placed per `na_position`.

    Supported `orders`:
        - If `data` is a Series: a list/tuple/ndarray/Index of ordered values.
        - If `data` is a DataFrame: a dict of {column_name: ordered_values}.

    Args:
        data (Union[pd.Series, pd.DataFrame]): Source data to be sorted.
        orders (Union[list, tuple, np.ndarray, pd.Index, dict]): Ordering rules.
        ascending (bool, optional): Order of the sort. Defaults to True.
        method (str, optional): Sorting method used by pandas. Defaults to 'mergesort'.
        na_position: Where to place undefined entries: 'first' or 'last'.
        validate (bool, optional): Raise if all values in a target become undefined.

    Returns:
        Union[pd.Series, pd.DataFrame]: Sorted data.

    Examples:
        # Series
        print(sort(df['i'], orders=['C', 'A', 'D']))

        # DataFrame with per-column orders
        print(sort(df, orders={'i': ['C', 'A', 'D']}))

        # Multiple columns with partial orders
        print(sort(df, orders={'a': [3, 2], 'b': [104, 102, 105, 100]}))
    """

    if not isinstance(data, (pd.DataFrame, pd.Series)):
        raise TypeError('`data` must be either a `pd.DataFrame` or `pd.Series` instance')

    def _unique_ordered(values):
        # Preserve order while dropping duplicates
        return list(dict.fromkeys(values).keys())

    if isinstance(data, pd.Series):
        if not isinstance(orders, (list, tuple, np.ndarray, pd.Index)):
            raise TypeError('For a `pd.Series`, `orders` must be a list-like of values')
        ordered_values = _unique_ordered(list(orders))
        cat_dtype = pd.CategoricalDtype(categories=ordered_values, ordered=True)
        data_ordered = data.astype(cat_dtype).sort_values(
            ascending=ascending,
            na_position=na_position,
        )
        if validate and data_ordered.isna().all():
            raise ValueError(
                'Every value in the series is now undefined! Are you sure you defined the `order` properly?'
            )
        return data.loc[data_ordered.index].copy()

    if not isinstance(orders, dict):
        raise TypeError('For a `pd.DataFrame`, `orders` must be a dict of column->order')

    data_ordered = data.copy()
    for col, col_order in orders.items():
        if col not in data_ordered.columns:
            raise KeyError(f'Column "{col}" does not exist in the dataframe')
        if not isinstance(col_order, (list, tuple, np.ndarray, pd.Index)):
            raise TypeError(f'Order for column "{col}" must be list-like')
        ordered_values = _unique_ordered(list(col_order))
        data_ordered[col] = pd.Categorical(data[col], categories=ordered_values, ordered=True)
        if validate and data_ordered[col].isna().all():
            raise ValueError(f'Every value in column "{col}" is now NaN! Have you defined the `order` properly?')

    data_ordered = data_ordered.sort_values(
        by=list(orders.keys()),
        ascending=ascending,
        kind=method,
        na_position=na_position,
    )
    return data.loc[data_ordered.index, :].copy()


def search(
    df: pd.DataFrame,
    value,
    *,  # indicates the end of the positional arguments. Later arguments can only be specified by keyword
    case: bool = True,
    regex: bool = False,
    tolerance: float | None = None,
):
    """
    Find positions of `value` in `df` and report them in a DataFrame.

    Note: 'row_number' is the positional index (0-based) in the DataFrame,
    not the row label. 'row_name' is the actual index label.

    Matching rules:
    - Numeric search (only numeric columns):
      * If `value` is NaN: match NaNs via isna; tolerance ignored.
      * If `value` is integer: exact equality.
      * If `value` is float:
          - Exact equality if `tolerance` is None
          - Otherwise match finite entries where |x - value| <= tolerance
            (NaN and ±Inf are ignored in tolerance comparisons).
    - String search (object/StringDtype columns only):
      * Exact, case-sensitive by default
      * Case-insensitive if case=False (using .str.casefold()).

    Parameters
    ----------
    df : pd.DataFrame
    value : Any
        Numeric (including NaN) or string. Other types return no matches.
    case : bool, default True
        Case sensitivity for string matches.
    regex : bool, default False
        Whether to interpret `value` as a regex pattern for string matches. The matching
        is performed using `pd.Series.str.contains()` (i.e., it does not need to match the entire string).
    tolerance : float | None, default None
        Absolute tolerance for float comparisons; ignored for ints, strings, and NaN.

    Returns
    -------
    pd.DataFrame
    """
    out_cols = ['value', 'row_name', 'col_name', 'row_number', 'col_number']
    if df.empty:
        return pd.DataFrame(columns=out_cols)

    # restricting to columns that can contain the value
    is_num = isinstance(value, (int, np.integer, float, np.floating))
    is_str = isinstance(value, str)
    assert is_num or is_str, 'Unsupported value type for search'
    if is_num:
        selected_cols = [
            c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) or 'mixed' in pd.api.types.infer_dtype(df[c])
        ]
    elif is_str:
        # selected_cols = [c for c in df.columns if pd.api.types.is_string_dtype(df[c]) or 'mixed' in pd.api.types.infer_dtype(df[c])]
        selected_cols = df.select_dtypes('object').columns.tolist() + df.select_dtypes('string').columns.tolist()
    if len(selected_cols) == 0:
        return pd.DataFrame(columns=out_cols)
    sub = df[selected_cols]

    # performing the search
    if is_num:
        # NaN search: match NaNs
        if isinstance(value, float) and np.isnan(value):
            mask = sub.isna()
        else:
            # Integer or no tolerance: exact equality
            if isinstance(value, (int, np.integer)) or tolerance is None:
                mask = sub.eq(value)
            else:
                # Float with tolerance: |x - value| <= tolerance on finite entries only
                # Build mask efficiently using NumPy, then convert to DataFrame
                mask_array = np.zeros(sub.shape, dtype=bool)
                for c_idx, c_name in enumerate(sub.columns):
                    if 'mixed' in pd.api.types.infer_dtype(sub[c_name]):
                        arr = pd.to_numeric(sub[c_name], errors='coerce').to_numpy(copy=False)
                    else:
                        arr = sub[c_name].to_numpy(copy=False)
                    is_finite = np.isfinite(arr)
                    mask_array[is_finite, c_idx] = np.abs(arr[is_finite] - value) <= tolerance
                mask = pd.DataFrame(mask_array, index=sub.index, columns=sub.columns)

    elif is_str:
        if regex:
            # Regex match
            mask = sub.apply(lambda col: col.str.contains(value, na=False, case=case), axis=0)
        elif case:
            mask = sub.eq(value)
        else:
            # Case-insensitive using casefold; preserves NaNs
            # Build mask efficiently using NumPy, then convert to DataFrame
            mask_array = np.zeros(sub.shape, dtype=bool)
            value_cf = value.casefold()
            for c_idx, c_name in enumerate(sub.columns):
                col_cf = sub[c_name].str.casefold()
                mask_array[:, c_idx] = col_cf.eq(value_cf).fillna(False).values
            mask = pd.DataFrame(mask_array, index=sub.index, columns=sub.columns)

    # Vectorized extraction of matches using np.where
    row_indices, col_indices = np.where(mask.values)
    if len(row_indices) == 0:
        return pd.DataFrame(columns=out_cols)

    return pd.DataFrame(
        {
            'value': list(sub.iat[ri, ci] for ri, ci in zip(row_indices, col_indices)),  # to keep the dtype intact
            'row_name': sub.index[row_indices],
            'col_name': sub.columns[col_indices],
            'row_number': row_indices,
            'col_number': col_indices,
        }
    )


def reorder_by_similarity(
    df: pd.DataFrame,
    axis: Literal['both', 'rows', 'columns'] = 'both',
    row_kws=None,
    col_kws=None,
) -> pd.DataFrame:
    """Reorder rows and columns of a numeric DataFrame by hierarchical clustering,
    similar to what ``sns.clustermap()`` does internally.
    This is useful for visualizations like heatmaps, where you want to group
    similar rows and columns together.

    All columns must be numeric (int, float, bool). Non-numeric columns
    (e.g. strings, datetimes) will raise a ``TypeError``.

    Args:
        df (pd.DataFrame): A numeric DataFrame with no NaN values.
        axis (Literal['both', 'rows', 'columns'], optional): Which axes to
            reorder. ``'both'`` (default) clusters rows and columns,
            ``'rows'`` clusters rows only, ``'columns'`` clusters columns only.
        row_kws (dict, optional): Keyword arguments forwarded to
            ``scipy.spatial.distance.pdist`` and ``scipy.cluster.hierarchy.linkage``
            for row clustering. Recognised keys are ``method`` (default ``'average'``)
            and ``metric`` (default ``'euclidean'``). Must be ``None`` when
            ``axis='columns'``.
        col_kws (dict, optional): Same as ``row_kws`` but for column clustering.
            Must be ``None`` when ``axis='rows'``.

    Returns:
        pd.DataFrame: A copy of ``df`` with rows and/or columns reordered according
        to the hierarchical clustering leaf order.

    Examples:
        >>> import numpy as np, pandas as pd
        >>> rng = np.random.default_rng(0)
        >>> df = pd.DataFrame(rng.standard_normal((5, 4)), columns=list('ABCD'))
        >>> reorder_by_similarity(df)
        >>> reorder_by_similarity(df, axis='rows')
        >>> reorder_by_similarity(df, axis='columns')
        >>> reorder_by_similarity(df, row_kws={'method': 'ward', 'metric': 'euclidean'})
        >>> reorder_by_similarity(df, col_kws={'metric': 'correlation'})
    """
    from scipy.cluster import hierarchy
    from scipy.spatial.distance import pdist

    _valid_axes = ('both', 'rows', 'columns')
    if axis not in _valid_axes:
        raise ValueError(f'`axis` must be one of {_valid_axes}, got {axis!r}')

    cluster_rows = axis in ('both', 'rows')
    cluster_cols = axis in ('both', 'columns')

    # Validate that *_kws are not provided for the unused axis
    if not cluster_rows and row_kws is not None:
        raise ValueError("`row_kws` must be None when axis='columns'")
    if not cluster_cols and col_kws is not None:
        raise ValueError("`col_kws` must be None when axis='rows'")

    # All columns must be numeric
    non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if len(non_numeric) > 0:
        raise TypeError(f'All columns must be numeric, but the following are not: {non_numeric}')

    # Check for NaN values and raise an error if any are found
    assert df.notna().all().all(), (
        'DataFrame contains NaN values! Please handle them before reordering (e.g., by imputation or dropping).'
    )

    # Cluster rows
    if cluster_rows and df.shape[0] >= 2:
        row_kws = {'method': 'average', 'metric': 'euclidean', **(row_kws or {})}
        row_dist = pdist(df, metric=row_kws['metric'])
        row_linkage = hierarchy.linkage(row_dist, method=row_kws['method'])
        row_order = hierarchy.leaves_list(row_linkage)
    else:
        row_order = np.arange(df.shape[0])

    # Cluster columns
    if cluster_cols and df.shape[1] >= 2:
        col_kws = {'method': 'average', 'metric': 'euclidean', **(col_kws or {})}
        col_dist = pdist(df.T, metric=col_kws['metric'])
        col_linkage = hierarchy.linkage(col_dist, method=col_kws['method'])
        col_order = hierarchy.leaves_list(col_linkage)
    else:
        col_order = np.arange(df.shape[1])

    # Reorder the DataFrame
    # equivalent to: df.iloc[np.ix_(row_order, col_order)], but more readable
    df_reordered = df.iloc[row_order, :].iloc[:, col_order].copy()

    return df_reordered


def _is_jupyter():
    """Detect if the code is running inside a Jupyter environment."""
    try:
        shell = get_ipython().__class__.__name__
        return shell in ('ZMQInteractiveShell', 'Shell')  # Jupyter Notebook/Lab/Colab
    except NameError:
        return False  # Plain Python interpreter


def _render(df, title):
    """Render a DataFrame using display() in Jupyter or print() otherwise."""

    print(f'===  {title}')

    if _is_jupyter():
        from IPython.display import display

        display(df)
    else:
        print(df.to_string())


def describe(df, n_top=3, show_numeric=True, show_categorical=True, return_dfs=False):
    """
    Enhanced pandas.DataFrame.describe() that splits the summary into two focused tables:
    - Numeric table: standard describe() stats + dtype
    - Categorical table: count, unique, top_n/count_n pairs + dtype

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to summarize.
    n_top : int
        Number of top frequent elements to show for categorical/string columns.
    show_numeric : bool
        Whether to display the numeric summary table.
    show_categorical : bool
        Whether to display the categorical/string summary table.

    Returns
    -------
    tuple of (numeric_df, categorical_df) — either may be None if not requested
    or if no columns of that type exist.
    """

    # Conflict check: return_dfs=True and show_*=True are mutually exclusive.
    if return_dfs and (show_numeric or show_categorical):
        raise ValueError(
            'If `return_dfs` is True, both `show_numeric` and `show_categorical` must be False '
            'to avoid displaying the tables.'
        )
    # If return_dfs is True, we won't display the tables, but will still compute them and return as DataFrames.
    if return_dfs:
        show_numeric = False
        show_categorical = False

    # ------------------------------------------------------------------ #
    #  Numeric Table                                                     #
    # ------------------------------------------------------------------ #
    numeric_df = None

    if show_numeric:
        numeric_cols = df.select_dtypes(include='number').columns.tolist()

        if numeric_cols:
            numeric_df = df[numeric_cols].describe().T
            numeric_df.insert(0, 'dtype', df[numeric_cols].dtypes)
            numeric_df = numeric_df.assign(
                count=lambda num_df: num_df['count'].astype(int),  # .fillna(0)
            )
        else:
            print('No numeric columns found.')

    # ------------------------------------------------------------------ #
    #  Categorical Table                                                 #
    # ------------------------------------------------------------------ #
    categorical_df = None

    if show_categorical:
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        if categorical_cols:
            rows = []

            for col in categorical_cols:
                series = df[col]
                value_counts = series.value_counts(dropna=False)

                row = {
                    'dtype': series.dtype,
                    'count': series.count(),
                    'n_unique': series.nunique(),
                }

                # Gracefully handle fewer unique values than n
                for i in range(1, n_top + 1):
                    if i <= len(value_counts):
                        row[f'top_{i}'] = value_counts.index[i - 1]
                        row[f'count_{i}'] = value_counts.iloc[i - 1]
                    else:
                        row[f'top_{i}'] = np.nan
                        row[f'count_{i}'] = np.nan

                rows.append(row)

            categorical_df = pd.DataFrame(rows, index=categorical_cols)

            # cast the count columns to `Int64` to allow `NaNs` while keeping integers
            count_cols = [f'count_{i}' for i in range(1, n_top + 1)]
            categorical_df[count_cols] = categorical_df[count_cols].astype('Int64')

        else:
            print('No categorical/string columns found.')

    # ------------------------------------------------------------------ #
    #  Display                                                             #
    # ------------------------------------------------------------------ #
    if show_numeric and numeric_df is not None:
        _render(numeric_df, 'NUMERIC SUMMARY')

    if show_categorical and categorical_df is not None:
        _render(categorical_df, 'CATEGORICAL SUMMARY')

    if return_dfs:
        return numeric_df, categorical_df
    else:
        return None


if __name__ == '__main__':
    pass
