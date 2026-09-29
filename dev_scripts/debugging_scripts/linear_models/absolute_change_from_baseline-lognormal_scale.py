# absolute change from baseline, response scale = log-normal

import numpy as np
import pandas as pd

from aa_utilities.computation.modeling import LinearModel
from aa_utilities.wrappers import RSpace

# initializations
rng = np.random.default_rng(seed=42)
n = 1000
n_visit = 5
ci = 0.95
pseudocount = 1e-20

# data generation
data = (
    pd.DataFrame()
    .assign(
        subject_idx=np.repeat(range(n // n_visit), n_visit),
        USUBJID=lambda df: df.subject_idx.map(lambda i: f'S{i:04d}'),
        TRT01P=lambda df: np.where(df.subject_idx % 2 == 0, 'Placebo', 'Treatment'),
        VISIT_idx=np.tile(range(n_visit), reps=n // n_visit),
        AVISIT=lambda df: df.VISIT_idx.map(lambda vi: f'Week {vi}'),
        BASE=rng.normal(loc=100, scale=25, size=n).astype(int),
    )
    .assign(
        # the above are all the same as absolute change from baseline, with the following modifications:
        BASE=np.exp(rng.normal(loc=0, scale=0.1, size=n)), # or rng.lognormal(mean=0, sigma=1)
        AVAL=lambda df: df.BASE + df.VISIT_idx * np.where(df.TRT01P == 'Placebo', 10, 30),
        # Week 1: Placebo = BASE + BASE * 10 = BASE * 11 + epsilon ;
        LOG_AVAL=lambda df: np.log(df.AVAL + pseudocount),
    )
)

R = RSpace()
model = LinearModel(space=R)
model.set_data(df=data, remove_categories=True, factorize=True)
model.fit_lm(formula=f'LOG_AVAL ~ TRT01P * AVISIT', ci=ci)
model.add_emmeans(spec=f'TRT01P * AVISIT', scale='link', ci=ci)
print(
    np.exp(model.results.ls_means[['estimate']])
    .sub(pseudocount)
    .round(2)
)

#                   estimate
# TRT01P    AVISIT          
# Placebo   Week 0      0.98
# Treatment Week 0      0.99
# Placebo   Week 1     10.99
# Treatment Week 1     31.00
# Placebo   Week 2     20.98
# Treatment Week 2     61.01
# Placebo   Week 3     30.99
# Treatment Week 3     91.01
# Placebo   Week 4     40.98
# Treatment Week 4    121.01