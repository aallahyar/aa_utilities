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
# Simulate a small clinical trial dataset

# - Binary treatment: 0 = Placebo, 1 = Drug
n_per_arm = 150
treatment = np.concatenate([
    np.zeros(n_per_arm, dtype=int),
    np.ones(n_per_arm, dtype=int),
])

# True parameters
# - True data-generating process: logit(P(response)) = beta0 + beta1 * treatment
#   where beta0 = logit(0.20) and beta1 = log(2.5) meaning the drug multiplies odds by 2.5
p0 = 0.20                     # baseline (placebo) response probability
beta0 = logit(p0)             # intercept on logit scale
OR_drug = 2.5                 # odds ratio for drug vs placebo
beta1 = np.log(OR_drug)       # slope on logit scale

# Generate individual probabilities and outcomes
lin_pred = beta0 + beta1 * treatment # linear predictor (in logit space)
p_response = expit(lin_pred)   # inverse-logit to get probabilities
response = rng.binomial(n=1, p=p_response)

#%%
# Assemble a labeled DataFrame
data = pd.DataFrame({
    "treatment": pd.Categorical(
        np.where(treatment == 0, "Placebo", "Drug"),
        categories=["Placebo", "Drug"]
    ),
    "response": response.astype(int),
})

# Optional quick checks:
print(data.head())
print(data['treatment'].value_counts())
print(data.groupby('treatment')['response'].mean()) # probability of responses (average response rates) by treatment arm

# %%

# Fit logistic regression (binomial with logit link)
ci = 0.95
model = LinearModel(space=R)
model.set_data(df=data, factorize=True, remove_categories=True)
model.set_reference({'treatment': 'Placebo'})
model.fit_logistic(
    formula="response ~ treatment",
    ci=ci,
)
model.add_emmeans(spec=f'treatment', scale='response', ci=ci)
model.results

#%%
R('''
# 3) Convert coefficients to odds ratios with 95% CI
est <- coef(fit)
se  <- sqrt(diag(vcov(fit)))

OR_table <- data.frame(
  Term = names(est),
  coef = est,
  OR = exp(est),
  CI_lower = exp(est - 1.96 * se),
  CI_upper = exp(est + 1.96 * se)

)
print(OR_table)
''')
# %%
R('''
# Predicted probabilities by arm
newdat <- data.frame(treatment = factor(c("Placebo","Drug"), levels = c("Placebo","Drug")))
pred <- predict(fit, newdata = newdat, type = "response", se.fit = TRUE)

pred_table <- data.frame(
  treatment = newdat$treatment,
  prob = pred$fit,
  lower = plogis(qlogis(pred$fit) - 1.96 * pred$se.fit),
  upper = plogis(qlogis(pred$fit) + 1.96 * pred$se.fit)
)
print(pred_table)
''')