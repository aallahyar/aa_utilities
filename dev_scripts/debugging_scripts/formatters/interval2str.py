import numpy as np
import pandas as pd

from aa_utilities.helpers import (
    formatters,
)

interval2str = formatters.interval2str

print(interval2str(pd.Interval(0.123456, 0.654321)))
print(interval2str(pd.Interval(1, 2, closed='both')))
print(interval2str(pd.Interval(1, 2, closed='left')))
print(interval2str(pd.Interval(1, 2, closed='right')))
print(interval2str(pd.Interval(1, 2, closed='neither')))

print(interval2str(pd.Interval(0.123456, 0.654321), fmt='{:.3f} <-> {:.3f}'))

print(interval2str(np.nan))

intv = pd.IntervalIndex.from_tuples([(0, 1), (1, 2), (2, 3)], closed='left')
print(interval2str(intv, fmt='{:.0f}, {:.3f}'))


print(interval2str(
    pd.Interval(1.5, 2.5), 
    fmt=lambda l, r: f"{l:.2f}–{r:.2f}",
))

