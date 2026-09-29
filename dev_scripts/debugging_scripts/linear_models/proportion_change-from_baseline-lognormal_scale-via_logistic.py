# proportion change from baseline, response scale = binary, modeled via logistic regression

import numpy as np
import pandas as pd

from aa_utilities.computation.modeling import LinearModel
from aa_utilities.wrappers import RSpace

# initializations
rng = np.random.default_rng(seed=42)
n = 10000
n_visit = 5
CI = 0.95

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
        BASE=None, # it does not matter for the logistic regression model
        AVAL=lambda df: df.apply(lambda row: rng.choice([0, 1], p=[
            1 - (row.VISIT_idx + 1) * (0.10 if row.TRT01P == 'Placebo' else 0.15),
                (row.VISIT_idx + 1) * (0.10 if row.TRT01P == 'Placebo' else 0.15),
        ], size=None), axis=1),
    )
)

# data.iloc[:10]

R = RSpace()
model = LinearModel(space=R)
model.set_data(df=data, remove_categories=True, factorize=True)
model.fit_logistic(formula=f'AVAL ~ TRT01P * AVISIT', ci=CI)

# Must be scale='response' to receive `prob` estimates; 
# since fit_logistic declares family = binomial(link = "logit") and we need to back-transform to the response scale
model.add_emmeans(spec=f'TRT01P:AVISIT', scale='response', ci=CI)

# returns odds ratios, not probability differences (e.g. "Placebo Week 0 / Treatment Week 0")
model.add_contrasts(method='pairwise', ci=CI)

print(
    model.results.ls_means,
    model.results.contrasts,
)

#                    prob  std.error   df  conf.low  conf.high  null  statistic       p.value
# TRT01P    AVISIT                                                                           
# Placebo   Week 0  0.113   0.010011  inf  0.094814   0.134157   0.5 -20.628754  1.515051e-94
# Treatment Week 0  0.145   0.011134  inf  0.124510   0.168214   0.5 -19.756532  7.048357e-87
# Placebo   Week 1  0.207   0.012812  inf  0.183011   0.233236   0.5 -17.208052  2.310754e-66
# Treatment Week 1  0.296   0.014436  inf  0.268511   0.325053   0.5 -12.507207  6.817970e-36
# Placebo   Week 2  0.308   0.014599  inf  0.280146   0.337326   0.5 -11.817837  3.157068e-32
# Treatment Week 2  0.441   0.015701  inf  0.410491   0.471961   0.5  -3.722761  1.970559e-04
# Placebo   Week 3  0.388   0.015410  inf  0.358261   0.418597   0.5  -7.022573  2.178186e-12
# Treatment Week 3  0.599   0.015498  inf  0.568280   0.628961   0.5   6.219492  4.987656e-10
# Placebo   Week 4  0.496   0.015811  inf  0.465066   0.526964   0.5  -0.252980  8.002840e-01
# Treatment Week 4  0.760   0.013506  inf  0.732542   0.785465   0.5  15.567589  1.208790e-54       term  null.value  odds.ratio  std.error  ...  conf.high  null  statistic        p.value
# contrast                                                                               ...  
# Placebo Week 0 / Treatment Week 0    TRT01P*AVISIT         0.0    0.751195   0.100903  ...   0.977438   1.0  -2.129859   3.318322e-02
# Placebo Week 0 / Placebo Week 1      TRT01P*AVISIT         0.0    0.488043   0.061865  ...   0.625687   1.0  -5.659067   1.521978e-08
# Placebo Week 0 / Treatment Week 1    TRT01P*AVISIT         0.0    0.302995   0.036830  ...   0.384506   1.0  -9.823102   8.954458e-23
# Placebo Week 0 / Placebo Week 2      TRT01P*AVISIT         0.0    0.286227   0.034666  ...   0.362913   1.0 -10.328941   5.213302e-25
# Placebo Week 0 / Treatment Week 2    TRT01P*AVISIT         0.0    0.161483   0.019130  ...   0.203687   1.0 -15.391984   1.852643e-53
# Placebo Week 0 / Placebo Week 3      TRT01P*AVISIT         0.0    0.200944   0.023935  ...   0.253784   1.0 -13.472350   2.275086e-41
# Placebo Week 0 / Treatment Week 3    TRT01P*AVISIT         0.0    0.085285   0.010141  ...   0.107669   1.0 -20.702551   3.285141e-95
# Placebo Week 0 / Placebo Week 4      TRT01P*AVISIT         0.0    0.129450   0.015304  ...   0.163206   1.0 -17.293133   5.299223e-67
# Placebo Week 0 / Treatment Week 4    TRT01P*AVISIT         0.0    0.040230   0.005002  ...   0.051332   1.0 -25.842723  2.937402e-147
# Treatment Week 0 / Placebo Week 1    TRT01P*AVISIT         0.0    0.649688   0.077305  ...   0.820328   1.0  -3.624434   2.895950e-04
# Treatment Week 0 / Treatment Week 1  TRT01P*AVISIT         0.0    0.403351   0.045750  ...   0.503767   1.0  -8.004920   1.195443e-15
# Treatment Week 0 / Placebo Week 2    TRT01P*AVISIT         0.0    0.381028   0.043038  ...   0.475446   1.0  -8.542463   1.313911e-17
# Treatment Week 0 / Treatment Week 2  TRT01P*AVISIT         0.0    0.214969   0.023669  ...   0.266744   1.0 -13.962073   2.655951e-44
# Treatment Week 0 / Placebo Week 3    TRT01P*AVISIT         0.0    0.267499   0.029640  ...   0.332382   1.0 -11.900699   1.173604e-32
