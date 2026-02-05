# selectors.py
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import ElasticNet
import matplotlib.pyplot as plt

from aa_utilities.computation.modeling import stability_selection


def linear_feature_weights_selector(
    estimator: BaseEstimator,
    feature_names: list[str] | None,
) -> stability_selection.SelectorResult:
    """
    Default linear selector returning signed model-native coefficients.
    - If coef_ is 1D: use directly.
    - If coef_ is 2D: use the first row by default.
    Note: Modify for multi-target/class aggregation (e.g., L2 across rows) if needed downstream.
    """
    if not hasattr(estimator, "coef_"):
        raise RuntimeError("Estimator does not expose coef_; provide a custom selector.")
    coef = estimator.coef_
    if getattr(coef, "ndim", 1) > 1:
        weights = np.asarray(coef[0, :], dtype=float)
    else:
        weights = np.asarray(coef, dtype=float)

    # sample_idxs and meta are filled by the orchestrator
    return stability_selection.SelectorResult(
        feature_weights=weights, 
        failure=None, 
        meta={"origin": "coef"},
    )

# Generate synthetic data
# rng = np.random.default_rng(123)
rng = np.random.default_rng()
n, p = 81, 200  # odd n to demonstrate CPSS drop-one halves
X = rng.normal(size=(n, p))
true_w = np.zeros(p)
# support = rng.choice(p, size=12, replace=False)
# true_w[support] = rng.normal(loc=2.0, scale=0.5, size=support.shape[0])
support = np.arange(10)  # for consistent testing
true_w[support] = np.linspace(0, 2, num=support.shape[0])
y = X @ true_w + rng.normal(scale=1.0, size=n)

# Choose a subsampler
# subsampler = stability_selection.subsampling.StandardSampler(ratio=0.5)
# subsampler = stability_selection.subsampling.BootstrapSampler(replacement=True, ratio=0.5)
subsampler = stability_selection.subsampling.ComplementaryPairsSampler()
# For stratified example:
# groups = np.array(["A"] * 50 + ["B"] * 31)
# subsampler = stability_selection.subsampling.StratifiedSampler(groups=groups, ratio=0.5)

# Estimator factory with per-iteration seed
def enet_factory(iter_seed: int | None) -> ElasticNet:
    return ElasticNet(
        alpha=0.1,
        l1_ratio=0.7,
        fit_intercept=True,
        max_iter=5000,
        random_state=iter_seed if iter_seed is not None else 42,
    )

config = stability_selection.StabilitySelectionConfig(
    n_iterations=1000,
    random_state=7,
    n_jobs=4,
    verbose=1,
    cap_threads=True,
    omp_num_threads=1,
    mkl_num_threads=1,
    openblas_num_threads=1,
    numexpr_num_threads=1,
)

stabsel = stability_selection.StabilitySelector(
    estimator_factory=enet_factory,
    selector_fn=linear_feature_weights_selector,
    subsampler=subsampler,
    config=config,
)

run = stabsel.fit(X=X, y=y, feature_names=[f"#{i}: {true_w[support[i]]:0.1f}" if i in support else f"-" for i in range(p)])

print(f"Collected {len(run.results)} per-fit results.")
ok = sum(1 for r in run.results if r.failure is None)
print(f"Successful fits: {ok} / {len(run.results)}")
first_ok = next((r for r in run.results if r.failure is None), None)
if first_ok is not None:
    print(f"First success: subset size={first_ok.meta.get('subset_size')}, "
            f"weights shape={first_ok.feature_weights.shape}")

# fit ElasticNet on full data for comparison
full_enet = enet_factory(iter_seed=42)
full_enet.fit(X, y)
full_enet_coef = (
    pd.Series(
        data=full_enet.coef_,
        index=run.feature_names,
    )
    .sort_values(ascending=False)
)

# Plot stability selection frequencies vs. full-data ElasticNet coefficients
n_results = len(run.results)
n_bar = 50
fig, axes = plt.subplots(2, 1, figsize=(15, 8))
stab_sel_freq = (
    pd.DataFrame(
        data=(result.feature_weights for result in run.results),
        columns=run.feature_names,
    )
    .ne(0)  # Non-zero feature selections
    .sum(axis=0)
    .sort_values(ascending=False)
    .div(n_results)  # Normalize by number of subsamples
    .mul(100)  # Convert to percentage
)
axes[0].bar(
    x=np.arange(n_bar),
    height=stab_sel_freq.head(n_bar).values,
)
axes[0].set_xticks(ticks=np.arange(n_bar), labels=stab_sel_freq.index[:n_bar], fontsize=8)
axes[0].set_title("Stability selection frequencies")
axes[0].set_ylabel("Selection frequency (%)")

# show ElasticNet on full data for comparison
axes[1].bar(
    x=np.arange(n_bar),
    height=full_enet_coef.head(n_bar).values,
    color="orange",
)
axes[1].set_xticks(ticks=np.arange(n_bar), labels=full_enet_coef.index[:n_bar], fontsize=8)
axes[1].set_title("ElasticNet coefficients trained on full data")
axes[1].set_ylabel("Coefficient value")

for ax in axes:
    ax.set_xlim(-0.5, n_bar - 0.5)
    ax.xaxis.set_tick_params(rotation=90)
    ax.grid(axis="y")

fig.suptitle(f"Stability selection, (n_iterations={n_results})")
fig.subplots_adjust(hspace=0.5, bottom=0.2)


# show scatterplot of stability selection freq vs full-data coefficients
nonzero_coefs = (
    pd.DataFrame(
        data=(result.feature_weights for result in run.results),
        columns=run.feature_names,
    )
    .ne(0)  # Non-zero feature selections
    .sum(axis=0)
    .div(n_results)  # Normalize by number of subsamples
    .mul(100)  # Convert to percentage
)

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(
    x=full_enet.coef_[full_enet.coef_!=0],
    y=nonzero_coefs.values[full_enet.coef_!=0],
    alpha=0.7,
)
ax.set_xlabel("ElasticNet coefficient value (full data)")
ax.set_ylabel("Stability selection frequency (%)")
ax.set_title("Stability selection frequency vs. full-data ElasticNet coefficients")
ax.axhline(y=70, color="red", linestyle="--", label="70% selection frequency")
ax.axvline(x=0, color="gray", linestyle="--")
ax.legend()

plt.show()

