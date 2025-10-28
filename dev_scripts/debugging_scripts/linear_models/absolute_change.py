import numpy as np
import pandas as pd

from aa_utilities.computation.modeling import LinearModel
from aa_utilities.wrappers import RSpace

# initialization
rng = np.random.default_rng(seed=42)
n = 1000
n_visit = 5
ci = 0.95

# data simulation
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
        AVAL=lambda df: df.BASE + df.VISIT_idx * np.where(df.TRT01P == 'Placebo', 1, 3),
        CHG=lambda df: df.AVAL - df.BASE,
    )
)

R = RSpace()
model = LinearModel(space=R)
model.set_data(df=data, remove_categories=True, factorize=True)
model.fit_lm(formula=f'CHG ~ TRT01P * AVISIT', ci=ci)
model.add_emmeans(spec=f'TRT01P * AVISIT', scale='response', ci=ci)
print(model.results.ls_means[['estimate']].round(2))
#                   estimate
# TRT01P    AVISIT          
# Placebo   Week 0      -0.0
# Treatment Week 0       0.0
# Placebo   Week 1       1.0
# Treatment Week 1       3.0
# Placebo   Week 2       2.0
# Treatment Week 2       6.0
# Placebo   Week 3       3.0
# Treatment Week 3       9.0
# Placebo   Week 4       4.0
# Treatment Week 4      12.0

model.add_contrasts(method='revpairwise', ci=ci)
print(
    model.results.contrasts
    [['estimate', 'conf.low', 'conf.high']]
    .round(2)
    .iloc[:10]
)
#                                      estimate  conf.low  conf.high
# contrast                                                          
# Treatment Week 0 - Placebo Week 0         0.0       0.0        0.0
# Placebo Week 1 - Placebo Week 0           1.0       1.0        1.0
# Placebo Week 1 - Treatment Week 0         1.0       1.0        1.0
# Treatment Week 1 - Placebo Week 0         3.0       3.0        3.0
# Treatment Week 1 - Treatment Week 0       3.0       3.0        3.0
# Treatment Week 1 - Placebo Week 1         2.0       2.0        2.0
# Placebo Week 2 - Placebo Week 0           2.0       2.0        2.0
# Placebo Week 2 - Treatment Week 0         2.0       2.0        2.0
# Placebo Week 2 - Placebo Week 1           1.0       1.0        1.0
# Placebo Week 2 - Treatment Week 1        -1.0      -1.0       -1.0
