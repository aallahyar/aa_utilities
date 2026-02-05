from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Any, Tuple
import os

import numpy as np
from sklearn.base import BaseEstimator
from joblib import Parallel, delayed

from ._subsampling import BaseSampler


@dataclass
class SelectorResult:
    """
    Per-fit output collected by the orchestrator from the selector function and augmented with context.
    - feature_weights: signed array of length p (model-native coefficients or analogous weights).
    - sample_idxs: integer indices used in this sub-fit.
    - failure: None if fit/selection succeeded; otherwise an error message string.
    - meta: metadata such as iteration, subfit_id (0..), iter_seed, subset_size, and optional origin.
    """
    feature_weights: np.ndarray
    sample_idxs: np.ndarray | None = None
    failure: str | None = None
    meta: Dict[str, Any] = field(default_factory=dict)


SelectorFn = Callable[[BaseEstimator, List[str] | None], SelectorResult]
EstimatorFactory = Callable[[int | None], BaseEstimator]


@dataclass
class StabilitySelectionConfig:
    """
    Configuration for the orchestrator.
    - n_iterations: number of draws/iterations.
    - random_state: base seed used to derive per-iteration seeds deterministically.
    - n_jobs: parallel workers for sub-fits across all iterations.
    - verbose: if > 0, joblib will print progress info.
    - cap_threads: whether to cap threads for BLAS/OpenMP backends within the parallel block.
    - *_num_threads: thread caps applied only inside the parallel block and restored afterward.
    """
    n_iterations: int = 200
    random_state: int | None = 42
    n_jobs: int = 1
    verbose: int = 0
    cap_threads: bool = True
    omp_num_threads: int = 1
    mkl_num_threads: int = 1
    openblas_num_threads: int = 1
    numexpr_num_threads: int = 1


@dataclass
class StabilitySelectionRun:
    """
    Returned by the orchestrator's fit method.
    - results: flat list of per-fit results (CPSS contributes 2 per iteration; others contribute 1).
    - feature_names: optional list aligned to feature_weights length p.
    - config, n_samples, n_features: run-level metadata for auditability.
    """
    results: List[SelectorResult]
    feature_names: List[str] | None
    config: StabilitySelectionConfig
    n_samples: int
    n_features: int


class StabilitySelector:
    """
    Orchestrates stability selection with a pluggable subsampling strategy.
    - subsampler: any BaseSampler subclass instance (StandardSampler, BootstrapSampler, StratifiedSampler, ComplementaryPairsSampler).
    - selector_fn: extracts feature_weights from a fitted estimator (signed, length p).
    - estimator_factory: creates a fresh estimator per sub-fit and accepts an optional seed.
    - No aggregation: returns per-fit results only; post-processing remains external.
    """

    def __init__(
        self,
        estimator_factory: EstimatorFactory,
        selector_fn: SelectorFn,
        subsampler: BaseSampler,
        config: StabilitySelectionConfig | None = None,
    ):
        self.estimator_factory = estimator_factory
        self.selector_fn = selector_fn
        self.subsampler = subsampler
        self.config = config or StabilitySelectionConfig()

    # -------------------------
    # Deterministic seed utils
    # -------------------------
    def _derive_iteration_seed(self, base_seed: int | None, iteration: int) -> int | None:
        if base_seed is None:
            return None
        # 32-bit wrap to keep deterministic behavior across platforms
        return (base_seed + 0x9E3779B1 * (iteration + 1)) & 0xFFFFFFFF

    # -------------------------
    # Thread-cap utils (scoped)
    # -------------------------
    def _apply_thread_caps(self) -> Dict[str, str | None]:
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

    def _restore_thread_caps(self, prev: Dict[str, str | None]) -> None:
        for key, val in prev.items():
            if val is None:
                if key in os.environ:
                    del os.environ[key]
            else:
                os.environ[key] = val

    # -------------------------
    # Sub-fit execution task
    # -------------------------
    def _subfit_task(
        self,
        it: int,
        subfit_id: int,
        sample_idxs: np.ndarray,
        X: np.ndarray,
        y: np.ndarray | None,
        feature_names: List[str] | None,
        fit_params: Dict[str, Any] | None,
        iter_seed: int | None,
    ) -> SelectorResult:
        # Create a fresh estimator, passing the per-iteration seed if supported
        try:
            estimator = self.estimator_factory(iter_seed)
        except TypeError:
            estimator = self.estimator_factory(None)

        try:
            X_sub = X[sample_idxs, :]
            y_sub = y[sample_idxs] if y is not None else None

            if fit_params:
                estimator.fit(X_sub, y_sub, **fit_params)
            else:
                estimator.fit(X_sub, y_sub)

            sel_result = self.selector_fn(estimator, feature_names)

            # Basic shape validation
            n_features = X.shape[1]
            if sel_result.feature_weights is not None and len(sel_result.feature_weights) != n_features:
                raise ValueError("selector returned feature_weights of wrong length.")

            # Attach indices and meta
            sel_result.sample_idxs = sample_idxs
            sel_result.failure = None
            meta = sel_result.meta or {}
            meta.update(
                {
                    "iteration": it,
                    "subfit_id": subfit_id,
                    "iter_seed": iter_seed,
                    "subset_size": len(sample_idxs),
                }
            )
            sel_result.meta = meta

        except Exception as e:
            sel_result = SelectorResult(
                feature_weights=None,
                sample_idxs=sample_idxs,
                failure=str(e),
                meta={
                    "iteration": it,
                    "subfit_id": subfit_id,
                    "iter_seed": iter_seed,
                    "subset_size": len(sample_idxs),
                },
            )

        return sel_result

    # -------------------------
    # Public API
    # -------------------------
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        feature_names: List[str] | None = None,
        fit_params: Dict[str, Any] | None = None,
    ) -> StabilitySelectionRun:
        """
        Run stability selection using the provided subsampler strategy.
        - For single-fit samplers (standard/stratified/bootstrap), each iteration yields 1 result.
        - For CPSS, each iteration yields 2 results (complementary halves).
        """
        X = np.asarray(X)
        n_samples, n_features = X.shape

        if feature_names is not None and len(feature_names) != n_features:
            raise ValueError("feature_names length must match X.shape[1].")

        if y is not None:
            y = np.asarray(y)
            if y.shape[0] != n_samples:
                raise ValueError("y must have the same number of samples as X.")

        base_seed = self.config.random_state
        iterations = list(range(self.config.n_iterations))

        # Scoped thread caps only during parallel execution
        prev_env = self._apply_thread_caps()
        try:
            if self.config.n_jobs == 1:
                results: List[SelectorResult] = []
                for it in iterations:
                    iter_seed = self._derive_iteration_seed(base_seed, it)
                    rng = np.random.default_rng(iter_seed)
                    sample_idxs_collection = self.subsampler.draw(n=n_samples, rng=rng)
                    for sub_id, sample_idxs in enumerate(sample_idxs_collection):
                        res = self._subfit_task(
                            it=it,
                            subfit_id=sub_id,
                            sample_idxs=sample_idxs,
                            X=X,
                            y=y,
                            feature_names=feature_names,
                            fit_params=fit_params,
                            iter_seed=iter_seed,
                        )
                        results.append(res)
            else:
                # Build tasks for all sub-fits across all iterations
                tasks: List[Tuple[int, int, np.ndarray, int | None]] = []
                for it in iterations:
                    iter_seed = self._derive_iteration_seed(base_seed, it)
                    rng = np.random.default_rng(iter_seed)
                    sample_idxs_collection = self.subsampler.draw(n=n_samples, rng=rng)
                    for sub_id, sample_idxs in enumerate(sample_idxs_collection):
                        tasks.append((it, sub_id, sample_idxs, iter_seed))

                if self.config.verbose > 0:
                    print(f"Starting stability selection: {len(tasks)} fits across {self.config.n_jobs} cores ...")
                job_pool = Parallel(
                    n_jobs=self.config.n_jobs,
                    backend="loky",
                    verbose=self.config.verbose > 0,
                )
                results = job_pool(
                    delayed(self._subfit_task)(
                        it=it,
                        subfit_id=sub_id,
                        sample_idxs=sample_idxs,
                        X=X,
                        y=y,
                        feature_names=feature_names,
                        fit_params=fit_params,
                        iter_seed=iter_seed,
                    )
                    for (it, sub_id, sample_idxs, iter_seed) in tasks
                )
        finally:
            self._restore_thread_caps(prev_env)

        return StabilitySelectionRun(
            results=results,
            feature_names=feature_names,
            config=self.config,
            n_samples=n_samples,
            n_features=n_features,
        )

