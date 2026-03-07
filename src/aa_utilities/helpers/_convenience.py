
from typing import (
    # Any, 
    # Callable, 
    Union, 
    # Literal,
)

import numpy as np
import pandas as pd


def quantile_cut(
        x: Union[pd.Series, np.ndarray, list],
        q: Union[int, list, np.ndarray],
        right=False,
        quantile_kwargs: dict = None,
        *args,
        **kwargs,
    ) -> pd.Categorical:
    """A convenience wrapper that enhances `pd.qcut` to allow left-inclusive intervals.

    Examples:
        x = pd.Series([0, 1, 2, 3, 4, np.nan, np.nan])
        res = quantile_cut(
            x=x,
            q=2,
            right=False,
            labels=['low', 'high'],
        )
        print(res)
    """

    # initializations
    if quantile_kwargs is None:
        quantile_kwargs = {}
    if isinstance(q, int):
        if q < 1:
            raise ValueError("`q` must be at least 1")
        probs = np.linspace(0.0, 1.0, q + 1)
    else:
        probs = np.array(q, dtype=float)
        if (probs < 0).any() or (probs > 1).any():
            raise ValueError('`q` values must be between 0 and 1')

    # Guard: empty or all-NaN input -> return all NaNs
    series = pd.Series(x)
    if series.dropna().size == 0:
        raise ValueError('Input `x` is empty or all-NaN!')
    
    # get quantile bins
    bins = (
        series
        .dropna()
        .quantile(
            q=probs,
            **quantile_kwargs,
        )
    )

    # adjust bins edges based on `right` argument, similar to what pd.cut does internally: 
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.cut.html
    range = series.max() - series.min()
    if right:
        bins.iat[ 0] -= range * 0.001
    else:
        bins.iat[-1] += range * 0.001

    # perform the quantile cut
    output = pd.cut(
        series,
        bins=bins,
        right=right,
        *args,
        **kwargs,
    )

    return output

if __name__ == '__main__':
    pass




