
from typing import (
    Any, 
    Callable, 
    Union, 
    Literal,
)

import numpy as np
import pandas as pd

def interval2str(interval, fmt='{:0.1f}, {:0.1f}'):
    """converting pd.Interval data type to a more readable string"""
    range_str = fmt.format(interval.left, interval.right)
    if interval.closed == 'both':
        output_str = '[' + range_str + ']'
    elif interval.closed == 'left':
        output_str = '[' + range_str + ')'
    elif interval.closed == 'right':
        output_str = '(' + range_str + ']'
    elif interval.closed == 'neither':
        output_str = '(' + range_str + ')'
    else:
        raise ValueError('Unknown bound')
    return output_str


def pvalue_to_asterisks(p_value):
    if p_value <= 0.0001:
        return '****'
    if p_value <= 0.001:
        return '***'
    if p_value <= 0.01:
        return '**'
    if p_value <= 0.05:
        return '*'
    return 'ns'


def generate_dataframe(n=100, seed=42):
    import datetime

    rng = np.random.default_rng(seed=seed)
    
    df = pd.DataFrame({
        'A' : rng.normal(loc=0, scale=1, size=n),
        'B' : rng.uniform(low=0, high=1, size=n),
        'C' : rng.integers(low=0, high=100, size=n),
        'D' : rng.exponential(scale=1, size=n),
        'E' : rng.choice(['flour', 'egg', 'oil', 'milk', 'water', 'salt', 'suger'], size=n),
        'F' : rng.choice([f'str_{i}' for i in range(10)], size=n),
        'G' : rng.choice(pd.date_range(
            datetime.datetime(2023,1,1),
            datetime.datetime(2024,1,1)
        ), size=n),
    })
    return df


def get_logger(name=None, level=None):
    """
    Defines a logger with a specified name, and level.
    if `name` is None, return the root logger of the hierarchy
    source: https://docs.python.org/3/library/logging.html#logging.getLogger
    """
    # import time
    import logging

    # initializations
    if level is None:
        level = logging.INFO

    logger = logging.getLogger(name=name)
    if len(logger.handlers) == 0:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt=r'%(asctime)s %(name)s [%(levelname)-s]: %(message)s', 
            datefmt=r'%Y-%m-%d %H:%M:%S',
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    logger.setLevel(level)
    logger.last_print = 0
    return logger


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

def sort_by(
        data: Union[pd.Series, pd.DataFrame],
        orders: Union[list, dict, pd.Series, pd.DataFrame],
        ascending=True,
        method='mergesort',
        na_position: Literal['first', 'last'] = 'last',
        validate=True,
    ) -> Union[pd.DataFrame, pd.Series]:
    """\
    Sorts the given data according to provided orders. Any undefined value will be assumed
    as NaN and placed at the `na_position`. The undefined orders are preserved.

    Args:
        data (Union[pd.Series, pd.DataFrame]): Source data that is going to be sorted
        orders (Union[dict, pd.Series, pd.DataFrame]): the requested order. All columns defined 
        in `orders` will be used for sorting.
        ascending (bool, optional): Order of the sort. Defaults to True.
        na_position: Where to place undefined enties, can be 'first', or 'last'.

    Returns:
        Union[pd.Series, pd.DataFrame]: Sorted data
    
    Notes: 
        * If sorting a `pd.Series`, then `orders` can be any `Iterable`. Note that in 
            this case, if `orders` is a `pd.Series`, its `.values` will be used and not 
            its `.index`.
        * The order of repeated values is kept untouched (i.e., stable sort)
        * The order of undefined values is kept untouched (i.e., stable sort)

    Examples:
        # pd.Series or pd.DataFrames can be sorted, 
        print(sort_by(df.i, orders=['C', 'A', 'D']))
        print(sort_by(df, orders={'i': ['C', 'A', 'D']}))

        # orders defined by unknown values are allowed
        print(sort_by(df.i, orders=['c', 'B', 'a', 'A']))

        # Error: at least one value needs to be defined
        # print(sort_by(df.i, orders=['c', 'a']))

        # the orders can be defined in multiple partially-overlapping columns
        print(sort_by(df, orders={'a': [3, 2], 'b': [104, 102, 105, 100]}))
        
        # the priorities can also be a pd.Series 
        print(sort_by(df, orders=df.i.loc[[1, 0, 3]]))

        # the priorities can also be a dataframe (values will be used, not the index)
        print(sort_by(df, orders=df.loc[[1, 0, 3], ['a', 'c']]))
    """

    # prepare orders
    if isinstance(orders, (pd.Series, )):
        orders = {orders.name: orders.values.tolist()}
    elif isinstance(orders, (pd.DataFrame, )):
        col_orders = {}
        for col_name in list(orders.keys()):
            col_orders[col_name] = list(orders[col_name].unique())
        orders = col_orders
    
    # if data is `pd.Series`, only orders.keys() are used
    if isinstance(data, (pd.Series, )):
        # keep unique values only, preserves order
        orders = list(dict.fromkeys(orders).keys()) # dict with values == `None`
        
        cat_dtype = pd.CategoricalDtype(categories=orders, ordered=True)
        data_ordered = (
            data
            .astype(cat_dtype)
            .sort_values(ascending=ascending, na_position=na_position)
            .copy()
        )
        if validate and data_ordered.isna().all():
            raise ValueError(f'Every value the series is now undefined!')
        return data.loc[data_ordered.index].copy()
    
    # if data is `pd.DataFrame`
    elif isinstance(data, (pd.DataFrame, )):

        # assign orders and sort, per column
        assert isinstance(orders, (dict, )), 'For a `pd.DataFrame`, `orders` must be defined per column'
        data_ordered = data.copy()
        for col in orders.keys():
            data_ordered[col] = pd.Categorical(data[col], categories=orders[col], ordered=True)
            if validate and data_ordered[col].isna().all():
                raise ValueError(f'Every value in column "{col}" is now NaN! Have you defined the ordered properly?')
        data_ordered = (
            data_ordered
            .sort_values(by=list(orders.keys()), ascending=ascending, kind=method, na_position=na_position)
        )
        return data.loc[data_ordered.index, :].copy()

    # otherwise
    else:
        raise ValueError('Data must be either a `pd.Series` or a `pd.DataFrame`')

if __name__ == '__main__':

    import pandas as pd

    container = {}



