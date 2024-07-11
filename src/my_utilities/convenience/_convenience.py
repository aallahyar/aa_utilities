

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

