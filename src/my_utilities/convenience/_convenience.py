
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
    import numpy as np
    import pandas as pd

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
    import time
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
        logger.last_print = time.time()
        return logger


def is_true(condition, data, message='The condition is False!'):
    """Returns the data, if `condition` is True
    The `data` is designed as the second argument, as it reads more naturally. 
    Even though it requires `lambda` and therefore its more lengthy in this design.
    Even if it was designed as the first argument, the condition often uses the dateframe
    to perform the check. Which mean that we need the `lambda` function anyway.

    Args:
        data (any): The object to be returned, if the condition is True.
        condition (bool): The condition to be tested
        message (str, optional): Messsage to be shown
            if the condition is False. Defaults to 'Condition is False!'.

    Returns:
        any: The `data` as it is provided (i.e., no modification).
    
    Examples:
        (
            pd.DataFrame({
                'a': range(5),
                'b': range(100, 105),
                'c': range(200, 205),
            })
            .pipe(lambda df: is_true(True, df))
            .pipe(lambda df: is_true(df.b.gt(50).all(), df.iloc[:3]))
            .pipe(lambda df: is_true(df.c.ge(200).all(), df.iloc[1:]))
        )

        #    a    b    c
        # 1  1  101  201
        # 2  2  102  202

    """
    assert condition, message
    return data

