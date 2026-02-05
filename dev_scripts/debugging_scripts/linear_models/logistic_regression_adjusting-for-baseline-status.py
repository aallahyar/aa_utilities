#%%
import numpy as np
import pandas as pd
from scipy.special import (
    logit, # inverse logit function, logit(p) = log(p/(1-p))
    expit, # logistic function, expit(x) = 1/(1+exp(-x))
)

from aa_utilities.wrappers import RSpace
from aa_utilities.computation.modeling import (
    LinearModel,
)

# Set seed for reproducibility
rng = np.random.default_rng(123)
R = RSpace()

#%%
# Extended R script: simulate baseline responder status and fit adjusted week-52 logistic regression
n_per_arm = 150
treatment = np.concatenate([
    np.zeros(n_per_arm, dtype=int),
    np.ones(n_per_arm, dtype=int),
])

#%%
# 2) True parameters (simple, intuitive)
p0 = 0.20                     # week-52 probability for Placebo + Baseline Non-responder
beta0 = logit(p0)             # intercept on logit scale
OR_drug = 2.5                 # treatment effect (odds ratio) for drug vs placebo
beta1 = np.log(OR_drug)       # slope on logit scale
OR_baseline = 3.0             # baseline responder effect
beta2 = np.log(OR_baseline)   # slope on logit scale

#%%
# 3) Simulate baseline responder status (balanced across arms for clarity)
baseline_prev = 0.40        # 40% responders at baseline
baseline_responder = rng.binomial(n=1, p=baseline_prev, size=treatment.shape[0])

# 4) Generate week-52 probabilities and outcomes using the adjusted model
linpred = beta0 + beta1 * treatment + beta2 * baseline_responder
p_response_52 = expit(linpred)   # inverse-logit
response_52 = rng.binomial(n=1, p=p_response_52)

#%%
# 5) Assemble dataset
data = pd.DataFrame({
    "treatment": pd.Categorical(np.where(treatment == 0, "Placebo", "Drug"),
                                categories=["Placebo", "Drug"]),
    "baseline_responder": pd.Categorical(np.where(baseline_responder == 0, "No", "Yes"),
                                         categories=["No", "Yes"]),
    "response_52": response_52.astype(int)
})

# Optional quick checks:
# print(data.head())
# print(data['treatment'].value_counts())
# print(data.groupby(['treatment', 'baseline_responder'])['response_52'].mean())

#%%
# Fit (adjusted week-52) logistic regression (binomial with logit link)
ci = 0.95
model = LinearModel(space=R)
model.set_data(df=data, factorize=True, remove_categories=True)
model.set_reference({'treatment': 'Placebo'})
model.fit_logistic(
    formula="response_52 ~ treatment + baseline_responder",
    ci=ci,
)
model.add_emmeans(spec=f'treatment + baseline_responder', scale='response', ci=ci)
model.results

# 
# ■ fit_coefs: <DataFrame> (3, 6)
# │                              estimate  std.error  statistic       p.value  conf.low  conf.high
# │       term                                                                                    
# │       (Intercept)           -1.196128   0.211342  -5.659692  1.516449e-08 -1.623989  -0.793259
# │       treatmentDrug          0.621445   0.250293   2.482873  1.303277e-02  0.134309   1.117195
# │       baseline_responderYes  1.236580   0.258641   4.781058  1.743751e-06  0.734690   1.750340
# └
# exp(-1.196128) / (1 + exp(-1.196128)) = 23.1%, estimating p0 = 0.20 with some noise
# exp(0.621445) / (1 + exp(0.621445)) = 1.86, close to true OR of 2.5. Adjusted change in log‑odds for Drug vs Placebo, holding baseline status constant (or equally correct "averaging over baseline status").
# exp(1.236580) = 3.44, close to true OR of 3.0; Adjusted change in log‑odds for Baseline Responder vs Non‑Responder, averaging over treatment arms.
#