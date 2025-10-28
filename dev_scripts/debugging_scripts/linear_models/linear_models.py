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
        AVAL=lambda df: df.BASE + df.VISIT_idx * np.where(df.TRT01P == 'Placebo', 1, 3),
        CHG=lambda df: df.AVAL - df.BASE,
    )
)

R = RSpace()
model = LinearModel(space=R)
model.set_data(data, remove_categories=True, factorize=True)
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

# percentage change from baseline, response scale = linear        
# same as "absolute change from baseline", with the following modifications:
AVAL=lambda df: df.BASE + df.BASE * df.VISIT_idx / 10 * np.where(df.TRT01P == 'Placebo', 1, 3)
PRC_CHG=lambda df: (df.AVAL - df.BASE) * 100 / df.BASE
model.fit_lm(formula=f'PRC_CHG ~ TRT01P * AVISIT', ci=ci)

#                   estimate
# TRT01P    AVISIT          
# Placebo   Week 0      -0.0
# Treatment Week 0      -0.0
# Placebo   Week 1      10.0
# Treatment Week 1      30.0
# Placebo   Week 2      20.0
# Treatment Week 2      60.0
# Placebo   Week 3      30.0
# Treatment Week 3      90.0
# Placebo   Week 4      40.0
# Treatment Week 4     120.0

# percentage change from baseline, response scale = log-normal
pseudocount = 1e-20
BASE=rng.lognormal(mean=0, sigma=1, size=n)
AVAL=lambda df: df.BASE + df.BASE * df.VISIT
LOG_CHG=lambda df: np.log(df.AVAL + pseudocount) - np.log(df.BASE + pseudocount)
model.fit_lm(formula=f'LOG_CHG ~ TRT01P * AVISIT', ci=ci)
model.add_emmeans(spec=f'TRT01P * AVISIT', scale='response', ci=ci)
print(np.exp(model.results.ls_means[['estimate']]).sub(1).mul(100).round(2))

#                   estimate
# TRT01P    AVISIT          
# Placebo   Week 0      -0.0
# Treatment Week 0      -0.0
# Placebo   Week 1      10.0
# Treatment Week 1      30.0
# Placebo   Week 2      20.0
# Treatment Week 2      60.0
# Placebo   Week 3      30.0
# Treatment Week 3      90.0
# Placebo   Week 4      40.0
# Treatment Week 4     120.0

# absolute change from baseline, response scale = log-normal
pseudocount = 1e-20
BASE=np.exp(rng.normal(loc=0, scale=1, size=n)) # or rng.lognormal(mean, sigma)
AVAL=lambda df: df.BASE + df.VISIT_idx * np.where(df.TRT01P == 'Placebo', 10, 30)
LOG_AVAL=lambda df: np.log(df.AVAL + pseudocount)

model.fit_lm(formula=f'LOG_AVAL ~ TRT01P * AVISIT', ci=ci)
model.add_emmeans(spec=f'TRT01P * AVISIT', scale='link', ci=ci)
print(np.exp(model.results.ls_means[['estimate']]).sub(pseudocount).round(2))

#                   estimate
# TRT01P    AVISIT          
# Placebo   Week 0      0.98
# Treatment Week 0      1.00
# Placebo   Week 1     11.38
# Treatment Week 1     31.76
# Placebo   Week 2     21.66
# Treatment Week 2     61.58
# Placebo   Week 3     31.48
# Treatment Week 3     91.52
# Placebo   Week 4     41.31
# Treatment Week 4    121.56

# absolute change from baseline, response scale = log-normal, using MMRM
model.fit_mmrm(formula=f'LOG_AVAL ~ TRT01P * AVISIT + us(AVISIT | USUBJID)', ci=ci)

#                   estimate
# TRT01P    AVISIT          
# Placebo   Week 0      0.98
# Treatment Week 0      1.00
# Placebo   Week 1     11.38
# Treatment Week 1     31.76
# Placebo   Week 2     21.66
# Treatment Week 2     61.58
# Placebo   Week 3     31.48
# Treatment Week 3     91.52
# Placebo   Week 4     41.31
# Treatment Week 4    121.56