# percentage change from baseline, response scale = log-normal, modeled via MMRM

import numpy as np
import pandas as pd

from aa_utilities.computation.modeling import LinearModel
from aa_utilities.wrappers import RSpace

# initializations
rng = np.random.default_rng(seed=42)
n = 1000
n_visit = 5
CI = 0.95
pseudocount = 0.001 # should be negligible compared to the values in BASE and AVAL

# data generation
data = (
    pd.DataFrame()
    .assign(
        subject_idx=np.repeat(range(n // n_visit), repeats=n_visit),
        USUBJID=lambda df: df.subject_idx.map(lambda i: f'S{i:04d}'),
        TRT01P=lambda df: np.where(df.subject_idx % 2 == 0, 'Placebo', 'Treatment'),
        VISIT_idx=np.tile(range(n_visit), reps=n // n_visit),
        AVISIT=lambda df: df.VISIT_idx.map(lambda vi: f'Week {vi}'),
        # the above are all the same as absolute change from baseline, with the following modifications:
    )
    .assign(
        BASE=lambda df: df.groupby('subject_idx')['subject_idx'].transform(lambda _: rng.lognormal(mean=1, sigma=0.1, size=None)),
        AVAL=lambda df: df.BASE + df.BASE * df.VISIT_idx * np.where(df.TRT01P == 'Placebo', rng.normal(0.0, 0.01, size=len(df)), rng.normal(0.1, 0.01, size=len(df))),
        LOG_CHG=lambda df: np.log(df.AVAL + pseudocount) - np.log(df.BASE + pseudocount),
    )
)

# remove baseline rows (Week 0) from the data, as they have no change from baseline (model fails otherwise)
is_base = data['AVISIT'] == 'Week 0'
data = data.loc[~is_base, :]

# data.iloc[:10]

R = RSpace()
model = LinearModel(space=R)
model.set_data(df=data, remove_categories=True, factorize=True)
model.fit_mmrm(formula=f'LOG_CHG ~ TRT01P * AVISIT + us(AVISIT | USUBJID)', ci=CI)

# type="response" and type="link" should give identical results. See the note in the LinearModel.add_emmeans() docstring
model.add_emmeans(spec=f'AVISIT:TRT01P', scale='link', ci=CI)
model.add_contrasts(method='pairwise', ci=CI)
print(
    np.exp(model.results.ls_means[['estimate']])
    .sub(1)
    .mul(100)
    .round(2)
)

#                   estimate
# AVISIT TRT01P             
# Week 1 Placebo        0.06
# Week 2 Placebo        0.15
# Week 3 Placebo       -0.38
# Week 4 Placebo       -0.61
# Week 1 Treatment      9.99
# Week 2 Treatment     19.81
# Week 3 Treatment     29.94
# Week 4 Treatment     40.18