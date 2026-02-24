from typing import (
    Any, 
    Callable, 
    Union, 
    Literal,
)

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype, infer_dtype


def generate_dataframe(n=100, seed=42):
    import datetime

    # initializations
    rng = np.random.default_rng(seed=seed)
    
    df = pd.DataFrame({
        'A': rng.normal(loc=0, scale=1, size=n),
        'B': rng.uniform(low=0, high=1, size=n),
        'C': rng.integers(low=0, high=100, size=n),
        'D': rng.exponential(scale=1, size=n),
        'E': rng.choice(['flour', 'egg', 'oil', 'milk', 'water', 'salt', 'suger'], size=n),
        'F': rng.choice([f'str_{i}' for i in range(10)], size=n),
        'G': rng.choice(pd.date_range(
            datetime.datetime(2023, 1, 1),
            datetime.datetime(2024, 1, 1)
        ), size=n),
    })
    return df


def is_true(data: Any, condition: Union[Callable, bool], message: Union[Callable, str] = 'The `condition` argument is False!'):
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
            .pipe(lambda df: is_true(data=df, True))
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
            dataframe
            .query(que_str)
            .copy(deep=True)
            .assign(**{
                indicator: que_id,
            })
        )

    merged = pd.concat(subsets, axis=0, ignore_index=False, sort=False, copy=True)

    return merged

def store(data: Any, namespace: dict, name: str='stored_data', copy=True) -> Any:
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
    
    if copy:
        namespace[name] = data.copy()
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
    *, # indicates the end of the positional arguments. Later arguments can only be specified by keyword
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
        selected_cols = [c for c in df.columns if is_numeric_dtype(df[c]) or 'mixed' in infer_dtype(df[c])]
    elif is_str:
        # selected_cols = [c for c in df.columns if is_string_dtype(df[c]) or 'mixed' in infer_dtype(df[c])]
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
                    if 'mixed' in infer_dtype(sub[c_name]):
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
    
    return pd.DataFrame({
        'value': list(sub.iat[ri, ci] for ri, ci in zip(row_indices, col_indices)), # to keep the dtype intact
        'row_name': sub.index[row_indices],
        'col_name': sub.columns[col_indices],
        'row_number': row_indices,
        'col_number': col_indices,
    })

if __name__ == '__main__':
    pass