#%%
import numpy as np
import pandas as pd

from aa_utilities.wrappers import RSpace
from aa_utilities.computation.modeling import (
    LinearModel,
)

#%%
# initialization
rng = np.random.default_rng(seed=42)
R = RSpace()
n_high = 1000
n_low = 1000
p_baseline_low = 0.20
p_baseline_high = 0.60

# Week 52 response rates
p_week52_low = p_baseline_low  # unchanged for Low
p_week52_high = np.clip(p_baseline_high + 0.20, 0.0, 1.0) # absolute increase of 0.20

#%%
# Assemble the subgroup data
df_low = pd.DataFrame({
    "ID": [f"L{idx+1:04d}" for idx in range(n_low)],
    "Group": "Low",
    "Baseline": rng.binomial(1, p_baseline_low, size=n_low),
    "Week52": rng.binomial(1, p_week52_low, size=n_low),
})
df_high = pd.DataFrame({
    "ID": [f"H{idx+1:04d}" for idx in range(n_high)],
    "Group": "High",
    "Baseline": rng.binomial(1, p_baseline_high, size=n_high),
    "Week52": rng.binomial(1, p_week52_high, size=n_high),
})

data = (
    pd.concat([df_low, df_high], ignore_index=True)
    .melt(id_vars=['ID', 'Group'], var_name='AVISIT', value_name='Response')
)

# Quick check: realized response rates
summary = (
    data
    .groupby(["Group", "AVISIT"])
    ["Response"]
    .mean()
    .unstack()
)
print(summary)

# data is the final dataset:
# You can now fit a logistic model with Group, AVISIT, and Group×AVISIT interaction.
ci = 0.95
model = LinearModel(space=R)
model.set_data(df=data, factorize=True, remove_categories=True)
model.set_reference({'Group': 'Low'})
model.fit_logistic(
    formula="Response ~ Group * AVISIT", # Without the interaction, the model cannot represent 
    # “High increases, Low stays the same at week 52". The interaction allows the effect of AVISIT to differ by Group.
    ci=ci,
)
model.add_emmeans(spec=f'Group * AVISIT', scale='response', ci=ci)
model.add_contrasts(method='pairwise', ci=ci)
model.add_contrasts(method='revpairwise', ci=ci, append=True) # append reverse contrasts
model.results
# %%
