from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Any, Dict, Tuple
import os
import numpy as np
from numpy.random import default_rng
from sklearn.base import BaseEstimator
from joblib import Parallel, delayed


# ---------- Result and Selector Types ----------

@dataclass
class SelectorResult:
    """
    Per-iteration outputs returned by the selector function.
    The wrapper does NOT aggregate these results.
    """
    score: Optional[np.ndarray]                        # shape (p,)
    feature_used: Optional[np.ndarray]                 # bool mask, shape (p,)
    sample_indices: np.ndarray                         # indices used to fit in this iteration
    failure: bool = False                              # True if estimator fit failed
    failure_info: Optional[str] = None                 # error message or traceback
    meta: Dict[str, Any] = field(default_factory=dict) # optional metadata (nnz, seed, etc.)


SelectorFn = Callable[[BaseEstimator, Optional[List[str]]], SelectorResult]
EstimatorFactory = Callable[[Optional[int]], BaseEstimator]  # accept per-iteration seed when supported


# ---------- Orchestrator ----------

@dataclass
class StabilitySelectionConfig:
    n_iterations: int = 200
    subsample_ratio: float = 0.5
    random_state: Optional[int] = 42
    method: str = "subsample"  # placeholder for future methods (e.g., "bootstrap")
    preserve_order: bool = True
    verbose: int = 0
    n_jobs: int = 1
    # Thread caps applied only within the parallel context
    cap_threads: bool = True
    omp_num_threads: int = 1
    mkl_num_threads: int = 1
    openblas_num_threads: int = 1
    numexpr_num_threads: int = 1


@dataclass
class StabilitySelectionRun:
    """
    Wrapper return object: strictly the list of per-iteration selector results and run-level metadata.
    """
    results: List[SelectorResult]
    feature_names: Optional[List[str]]
    config: StabilitySelectionConfig
    n_samples: int
    n_features: int


class StabilitySelector:
    """
    Model-agnostic stability selection orchestrator with parallel execution.

    - Subsamples data each iteration (deterministically seeded)
    - Instantiates a fresh estimator via estimator_factory (seed passed if supported)
    - Fits the estimator on the subsample
    - Calls the selector function to extract per-iteration outputs
    - Returns a list of SelectorResult objects (no aggregation)
    - Applies environment thread caps only within the parallel execution context
    """

    def __init__(
        self,
        estimator_factory: EstimatorFactory,
        selector_fn: SelectorFn,
        config: Optional[StabilitySelectionConfig] = None,
    ):
        self.estimator_factory = estimator_factory
        self.selector_fn = selector_fn
        self.config = config or StabilitySelectionConfig()

    def _derive_iteration_seed(self, base_seed: Optional[int], iteration: int) -> Optional[int]:
        if base_seed is None:
            return None
        # Simple deterministic derivation; avoids collisions
        return (base_seed + 0x9E3779B1 * (iteration + 1)) & 0xFFFFFFFF

    def _subsample_indices(self, n: int, rng: np.random.Generator) -> np.ndarray:
        k = max(1, int(round(self.config.subsample_ratio * n)))
        idx = rng.choice(n, size=k, replace=False)  # subsample without replacement
        if self.config.preserve_order:
            idx = np.sort(idx)
        return idx

    def _iteration_task(
        self,
        it: int,
        X: np.ndarray,
        y: Optional[np.ndarray],
        feature_names: Optional[List[str]],
        fit_params: Optional[Dict[str, Any]],
        base_seed: Optional[int],
    ) -> SelectorResult:
        """
        Executes a single stability selection iteration.
        Intended to run in parallel workers.
        """
        iter_seed = self._derive_iteration_seed(base_seed, it)
        rng = default_rng(iter_seed)
        idx = self._subsample_indices(n=X.shape[0], rng=rng)
        X_sub = X[idx]
        y_sub = y[idx] if y is not None else None

        # Instantiate estimator; pass per-iteration seed when supported
        try:
            estimator = self.estimator_factory(iter_seed)
        except TypeError:
            # Factory does not accept seed; fallback
            estimator = self.estimator_factory(None)

        try:
            if fit_params:
                estimator.fit(X_sub, y_sub, **fit_params)
            else:
                estimator.fit(X_sub, y_sub)

            # Selector provides scores and feature_used
            sel_result = self.selector_fn(estimator, feature_names)
            # Attach indices and metadata
            sel_result.sample_indices = idx
            sel_result.failure = False
            sel_result.failure_info = None

            # Basic sanity checks
            n_features = X.shape[1]
            if sel_result.score is not None and len(sel_result.score) != n_features:
                raise ValueError("selector returned score of wrong length.")
            if sel_result.feature_used is not None and len(sel_result.feature_used) != n_features:
                raise ValueError("selector returned feature_used of wrong length.")

            # Augment meta with iteration and seed
            meta = sel_result.meta or {}
            meta.update({"iteration": it, "iter_seed": iter_seed, "subsample_size": len(idx)})
            sel_result.meta = meta

        except Exception as e:
            sel_result = SelectorResult(
                score=None,
                feature_used=None,
                sample_indices=idx,
                failure=True,
                failure_info=str(e),
                meta={"iteration": it, "iter_seed": iter_seed, "subsample_size": len(idx)},
            )

        return sel_result

    def _apply_thread_caps(self) -> Dict[str, Optional[str]]:
        """
        Set environment thread caps; return previous values for restoration.
        Caps are applied only within the parallel execution context.
        """
        prev = {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
        }
        if self.config.cap_threads:
            os.environ["OMP_NUM_THREADS"] = str(self.config.omp_num_threads)
            os.environ["MKL_NUM_THREADS"] = str(self.config.mkl_num_threads)
            os.environ["OPENBLAS_NUM_THREADS"] = str(self.config.openblas_num_threads)
            os.environ["NUMEXPR_NUM_THREADS"] = str(self.config.numexpr_num_threads)
        return prev

    def _restore_thread_caps(self, prev: Dict[str, Optional[str]]) -> None:
        """
        Restore environment thread caps after parallel execution.
        """
        for key, val in prev.items():
            if val is None:
                if key in os.environ:
                    del os.environ[key]
            else:
                os.environ[key] = val

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        fit_params: Optional[Dict[str, Any]] = None,
    ) -> StabilitySelectionRun:
        """
        Parameters
        - X: array-like (n_samples, n_features) [pre-transformed if needed]
        - y: array-like (n_samples,) or (n_samples, n_targets), optional
        - feature_names: optional list of feature names length n_features
        - fit_params: optional dict passed to estimator.fit

        Returns
        - StabilitySelectionRun with list of SelectorResult per iteration (no aggregation)
        """
        X = np.asarray(X)
        n_samples, n_features = X.shape
        if feature_names is not None and len(feature_names) != n_features:
            raise ValueError("feature_names length must match X.shape[1].")

        if y is not None:
            y = np.asarray(y)
            if y.shape[0] != n_samples:
                raise ValueError("y must have same number of samples as X.")

        if not (0 < self.config.subsample_ratio <= 1.0):
            raise ValueError("subsample_ratio must be in (0, 1].")

        base_seed = self.config.random_state

        # Prepare iteration indices
        iterations = list(range(self.config.n_iterations))

        # Scoped thread caps for the parallel context
        prev_env = self._apply_thread_caps()

        try:
            if self.config.n_jobs == 1:
                # Serial execution
                results = [
                    self._iteration_task(
                        it=it,
                        X=X,
                        y=y,
                        feature_names=feature_names,
                        fit_params=fit_params,
                        base_seed=base_seed,
                    )
                    for it in iterations
                ]
            else:
                # Parallel execution with joblib Loky backend
                # Note: estimator_factory and selector_fn must be picklable
                results = Parallel(
                    n_jobs=self.config.n_jobs,
                    backend="loky",
                    verbose=self.config.verbose > 0,
                )(
                    delayed(self._iteration_task)(
                        it=it,
                        X=X,
                        y=y,
                        feature_names=feature_names,
                        fit_params=fit_params,
                        base_seed=base_seed,
                    )
                    for it in iterations
                )
        finally:
            # Restore environment variables to avoid affecting other parts of the pipeline
            self._restore_thread_caps(prev_env)

        return StabilitySelectionRun(
            results=results,
            feature_names=feature_names,
            config=self.config,
            n_samples=n_samples,
            n_features=n_features,
        )


# ---------- Example Selector Function (Linear Models) ----------

def linear_selector_scores(estimator: BaseEstimator, feature_names: Optional[List[str]]) -> SelectorResult:
    """
    Model-agnostic score extractor for linear estimators.
    - Uses coef_ if available (score = absolute value; sign omitted to keep 'score' generic)
    - Falls back to feature_importances_ if present
    - If neither is available, raises an informative error
    """
    score = None
    feature_used = None

    if hasattr(estimator, "coef_"):
        coef = estimator.coef_
        if hasattr(coef, "ndim") and coef.ndim > 1:
            coef_vec = np.linalg.norm(coef, axis=0)
        else:
            coef_vec = np.asarray(coef)
        score = np.abs(coef_vec)
        feature_used = np.ones_like(score, dtype=bool)
    elif hasattr(estimator, "feature_importances_"):
        imp = np.asarray(estimator.feature_importances_)
        score = imp.copy()
        feature_used = np.ones_like(score, dtype=bool)
    else:
        raise RuntimeError("Estimator does not expose coef_ or feature_importances_; provide a custom selector.")

    meta = {
        "nnz": int(np.sum(score > 0)),
    }

    return SelectorResult(
        score=score,
        feature_used=feature_used,
        sample_indices=np.array([], dtype=int),  # will be set by orchestrator
        failure=False,
        failure_info=None,
        meta=meta,
    )


# ---------- Usage Example ----------

if __name__ == "__main__":
    from sklearn.linear_model import ElasticNet

    rng = np.random.default_rng(123)
    n, p = 80, 200
    X = rng.normal(size=(n, p))
    true_w = np.zeros(p)
    support = rng.choice(p, size=10, replace=False)
    true_w[support] = rng.normal(loc=2.0, scale=0.5, size=10)
    y = X @ true_w + rng.normal(scale=1.0, size=n)

    # Estimator factory that accepts a seed (if estimator supports random_state)
    def enet_factory(iter_seed: Optional[int]) -> BaseEstimator:
        return ElasticNet(
            alpha=0.1,
            l1_ratio=0.7,
            fit_intercept=True,
            max_iter=5000,
            random_state=iter_seed if iter_seed is not None else 0,
        )

    config = StabilitySelectionConfig(
        n_iterations=100,
        subsample_ratio=0.5,
        random_state=7,   # base seed for deterministic subsampling and per-iteration seeds
        verbose=1,
        n_jobs=4,         # parallel across iterations
        cap_threads=True, # cap threads only within this parallel block
        omp_num_threads=1,
        mkl_num_threads=1,
        openblas_num_threads=1,
        numexpr_num_threads=1,
    )

    stabsel = StabilitySelector(
        estimator_factory=enet_factory,
        selector_fn=linear_selector_scores,
        config=config,
    )

    run = stabsel.fit(X=X, y=y, feature_names=[f"g{i}" for i in range(p)])

    print(f"Collected {len(run.results)} per-iteration selector outputs.")
    ok = sum(1 for r in run.results if not r.failure)
    print(f"Successful iterations: {ok} / {len(run.results)}")
    # Inspect one iteration
    first_ok = next((r for r in run.results if not r.failure), None)
    if first_ok:
        print(f"First success: nnz={first_ok.meta.get('nnz')}, subsample size={len(first_ok.sample_indices)}")

