# percentage change from baseline, response scale = linear        
import numpy as np
import pandas as pd

from aa_utilities.computation.modeling import LinearModel
from aa_utilities.wrappers import RSpace

rng = np.random.default_rng(seed=42)
n = 1000
n_visit = 5
ci = 0.95
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
        BASE=lambda df: df.groupby('USUBJID').BASE.transform(lambda g: g.iat[0]),

        # the above are all the same as absolute change from baseline, with the following modifications:
        AVAL=lambda df: df.BASE + df.BASE * df.VISIT_idx / 10 * np.where(df.TRT01P == 'Placebo', 1, 3),
        PRC_CHG=lambda df: (df.AVAL - df.BASE) * 100 / df.BASE,
    )
)

R = RSpace()
model = LinearModel(space=R)
model.set_data(df=data, remove_categories=True, factorize=True)
model.fit_lm(formula=f'PRC_CHG ~ TRT01P * AVISIT', ci=ci)
model.add_emmeans(spec=f'TRT01P * AVISIT', scale='response', ci=ci)
print(model.results.ls_means[['estimate']].round(2))
#                   estimate
# TRT01P    AVISIT          
# Placebo   Week 0      -0.0
# Treatment Week 0       0.0
# Placebo   Week 1     100.0
# Treatment Week 1     300.0
# Placebo   Week 2     200.0
# Treatment Week 2     600.0
# Placebo   Week 3     300.0
# Treatment Week 3     900.0
# Placebo   Week 4     400.0
# Treatment Week 4    1200.0
