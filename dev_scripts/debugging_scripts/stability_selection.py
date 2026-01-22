# selectors.py
from __future__ import annotations
from typing import Any
import numpy as np
from sklearn.base import BaseEstimator
from stability_selection import SelectorResult

from sklearn.linear_model import ElasticNet

from aa_utilities.computation.modeling.stability_selection import (
    _subsampling,
    StabilitySelector,
    StabilitySelectionConfig,
)


def linear_feature_weights_selector(
    estimator: BaseEstimator,
    feature_names: list[str] | None,
) -> SelectorResult:
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
    return SelectorResult(feature_weights=weights, sample_idxs=np.array([], dtype=int), failure=None, meta={"origin": "coef"})



rng = np.random.default_rng(123)
n, p = 81, 200  # odd n to demonstrate CPSS drop-one halves
X = rng.normal(size=(n, p))
true_w = np.zeros(p)
support = rng.choice(p, size=12, replace=False)
true_w[support] = rng.normal(loc=2.0, scale=0.5, size=support.shape[0])
y = X @ true_w + rng.normal(scale=1.0, size=n)

# Choose a subsampler
# subsampler = StandardSampler(ratio=0.5)
# subsampler = BootstrapSampler(replacement=True, ratio=0.5)
subsampler = _subsampling.ComplementaryPairsSampler()
# For stratified example:
# groups = np.array(["A"] * 50 + ["B"] * 31)
# subsampler = subsampling.StratifiedSampler(groups=groups, ratio=0.5)

# Estimator factory with per-iteration seed
def enet_factory(iter_seed: int | None) -> ElasticNet:
    return ElasticNet(
        alpha=0.1,
        l1_ratio=0.7,
        fit_intercept=True,
        max_iter=5000,
        random_state=iter_seed if iter_seed is not None else 0,
    )

config = StabilitySelectionConfig(
    n_iterations=20,
    random_state=7,
    n_jobs=4,
    verbose=1,
    cap_threads=True,
    omp_num_threads=1,
    mkl_num_threads=1,
    openblas_num_threads=1,
    numexpr_num_threads=1,
)

stabsel = StabilitySelector(
    estimator_factory=enet_factory,
    selector_fn=linear_feature_weights_selector,
    subsampler=subsampler,
    config=config,
)

run = stabsel.fit(X=X, y=y, feature_names=[f"g{i}" for i in range(p)])

print(f"Collected {len(run.results)} per-fit results.")
ok = sum(1 for r in run.results if r.failure is None)
print(f"Successful fits: {ok} / {len(run.results)}")
first_ok = next((r for r in run.results if r.failure is None), None)
if first_ok is not None:
    print(f"First success: subset size={first_ok.meta.get('subset_size')}, "
            f"weights shape={first_ok.feature_weights.shape}")

