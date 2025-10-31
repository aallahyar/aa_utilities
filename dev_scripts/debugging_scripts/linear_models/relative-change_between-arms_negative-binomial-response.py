# relative change when comparing arms

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from aa_utilities.computation.modeling import LinearModel
from aa_utilities.wrappers import RSpace

# initializations
rng = np.random.default_rng(seed=42)
ci = 0.95
n = 10000
n1 = int(n * 0.2)
n2 = int(n * 0.2)
n3 = int(n * 0.6)
theta = 20.0  # same theta across arms, with moderate overdispersion; higher = closer to Poisson

# generate negative binomial data, parameterized by mean and dispersion
def rnbinom(mu, theta, size):
    # NB parameterization: Var = mu + mu^2 / theta
    p = theta / (theta + mu)      # success prob
    r = theta                     # number of failures (size)
    return rng.negative_binomial(r, p, size=size)

# data generation
data = (
    pd.DataFrame()
    .assign(
        TRT01P=['Placebo'] * n1 + ['Treated18'] * n2 + ['Treated36'] * n3,
        EXACN=(
            rnbinom(2, theta, n1).tolist() 
            + rnbinom(1.8, theta, n2).tolist() 
            + rnbinom(3.6, theta, n3).tolist()
        ),
        # EXACN= (
        #     rng.poisson(lam=20, size=n1).tolist() 
        #     + rng.poisson(lam=18, size=n2).tolist() 
        #     + rng.poisson(lam=36, size=n3).tolist()
        # ),
        TMEXRISK=(
            rng.normal(100, 1, n1).astype(int).tolist() 
            + rng.normal(100, 1, n2).astype(int).tolist() 
            + rng.normal(200, 1, n3).astype(int).tolist()
        ),
    )
)

# modeling
R = RSpace()
model = LinearModel(space=R)
model.set_data(df=data, remove_categories=True, factorize=True)
model.fit_negbin(formula=f'EXACN ~ offset(log(TMEXRISK)) + TRT01P', exponentiate=True, ci=ci)
model.add_emmeans(spec=f'TRT01P', scale='response', emm_kws=', offset=log(100), rg.limit = 100000', ci=ci)

print(model.results)
# <Container> (7)
# ├─■ n_samples: 10000
# ├─■ model_name: 'negative_binomial'
# ├─■ formula: 'EXACN ~ offset(log(TMEXRISK)) + TRT01P'
# ├─■ n_observations: 10000
# ├─■ fit_theta: 19.467909892737037
# ├─■ fit_coefs: <DataFrame> (3, 6)
# │                        estimate  std.error   statistic   p.value  conf.low  conf.high
# │       term                                                                           
# │       (Intercept)      0.019819   0.016710 -234.653863  0.000000  0.019178   0.020476
# │       TRT01PTreated18  0.906728   0.024178   -4.049758  0.000051  0.864747   0.950717
# │       TRT01PTreated36  0.923205   0.018260   -4.375795  0.000012  0.890852   0.956959
# └─■ ls_means: <DataFrame> (3, 8)
#                    response  std.error   df  conf.low  conf.high  null  statistic        p.value
#         TRT01P                                                                                  
#         Placebo    1.981936   0.033118  inf  1.918077   2.047922   1.0  40.937703   0.000000e+00
#         Treated18  1.797076   0.031401  inf  1.736572   1.859688   1.0  33.545537  1.045857e-246
#         Treated36  1.829733   0.013473  inf  1.803517   1.856331   1.0  82.052743   0.000000e+00

# plotting exacerbation counts by treatment arm
fig = plt.figure(figsize=(4, 3))
ax = fig.gca()
sns.histplot(
    data=data,
    # x='TRT01P',
    x='EXACN',
    # x=np.log(data['EXACN'] + 1e-4),
    hue='TRT01P',
    binwidth=1.0,
    binrange=(0, 16),
    ax=ax,
)
# ax.set_xscale('symlog', linthresh=1e-4)
plt.show()


