from __future__ import annotations

from collections.abc import Callable

import numpy as np
from joblib import Parallel, delayed, effective_n_jobs
from scipy.spatial.distance import cdist

# Metrics whose C implementations release the GIL — threading is sufficient and
# avoids the process-spawn overhead that loky incurs.
_GIL_FREE_METRICS = frozenset({
    "braycurtis", "canberra", "chebyshev", "cityblock", "correlation",
    "cosine", "euclidean", "jensenshannon", "mahalanobis", "minkowski",
    "seuclidean", "sqeuclidean",
    "hamming", "jaccard", "matching", "rogerstanimoto",
    "russellrao", "sokalsneath", "yule",
})


def _compute_chunk(
    X_chunk: np.ndarray,
    Y: np.ndarray,
    metric: str | Callable,
    metric_kwargs: dict,
) -> np.ndarray:
    return cdist(X_chunk, Y, metric=metric, **metric_kwargs)


def pairwise_distances(
    X: np.ndarray,
    Y: np.ndarray | None = None,
    metric: str | Callable = "euclidean",
    n_jobs: int = 1,
    chunk_size: int | None = None,
    force_symmetry: bool = False,
    backend: str | None = None,
    **metric_kwargs,
) -> np.ndarray:
    """Compute pairwise distances between samples using multiple cores.

    Supports all-vs-all (square) and X-vs-Y (rectangular) modes.
    Row chunks of ``X`` are distributed across workers via :mod:`joblib`.

    Parameters
    ----------
    X : np.ndarray
        2-D array of shape ``(n_samples, n_features)``.
    Y : np.ndarray or None, default=None
        2-D array of shape ``(m_samples, n_features)``.  When ``None``, the
        all-vs-all distance matrix of ``X`` is computed (``n x n``).
    metric : str or callable, default="euclidean"
        Any metric accepted by :func:`scipy.spatial.distance.cdist`, e.g.
        ``"euclidean"``, ``"cosine"``, ``"correlation"``, ``"minkowski"``, or
        a user-defined ``callable(u, v) -> float``.
    n_jobs : int, default=1
        Number of parallel workers.  ``-1`` uses all available CPUs.
        Parallelism is achieved by splitting rows of ``X`` into chunks.
    chunk_size : int or None, default=None
        Number of rows of ``X`` per chunk.  When ``None``, the chunk size is
        chosen automatically so that the number of chunks equals the effective
        number of workers.
    force_symmetry : bool, default=False
        When ``True`` and ``Y is None``, the result is symmetrised via
        ``(D + D.T) / 2``, eliminating floating-point asymmetries that can
        arise from different evaluation orders.  Has no effect when ``Y`` is
        provided.
    backend : str or None, default=None
        joblib backend to use (``"threading"``, ``"loky"``, ``"multiprocessing"``).
        When ``None`` (recommended), the backend is chosen automatically:
        ``"threading"`` for built-in string metrics whose C implementations
        release the GIL; ``"loky"`` (true processes) for callables that hold
        the GIL.
    **metric_kwargs
        Extra keyword arguments forwarded to :func:`scipy.spatial.distance.cdist`
        (e.g. ``p=3`` for Minkowski distance).

    Returns
    -------
    np.ndarray
        Distance matrix of shape ``(n_samples, n_samples)`` when ``Y is None``,
        or ``(n_samples, m_samples)`` otherwise.

    Raises
    ------
    ValueError
        If ``X`` or ``Y`` are not 2-D, or if their feature dimensions differ.

    Notes
    -----
    **Why not** ``sklearn.metrics.pairwise_distances``?

    scikit-learn's implementation hard-codes ``backend="threading"`` for all
    parallel work (see ``sklearn.metrics.pairwise._parallel_pairwise``).
    Threading only helps when the underlying C code releases the GIL.  For
    built-in scipy metrics this is true, but for any **custom callable metric**
    the GIL is held throughout and the extra threads add overhead without
    providing any parallelism — benchmarks show ~2.5–3× *slower* results vs.
    true multi-processing at n ≥ 500 samples.

    This implementation selects the backend automatically:

    * ``"threading"`` — for known GIL-free string metrics (euclidean, cosine,
      etc.) where BLAS/C kernels release the GIL and cross-process memory
      copies would waste more time than they save.
    * ``"loky"`` — for callable metrics (GIL is held), giving true multi-core
      parallelism via process workers.

    The compute kernel is :func:`scipy.spatial.distance.cdist`, which is a
    compiled C routine and imposes essentially zero overhead vs. calling scipy
    directly at ``n_jobs=1``.

    Examples
    --------
    >>> import numpy as np
    >>> from aa_utilities.computation import pairwise_distances
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((100, 20))
    >>> D = pairwise_distances(X, metric="euclidean", n_jobs=2)
    >>> D.shape
    (100, 100)
    >>> Y = rng.standard_normal((60, 20))
    >>> D_xy = pairwise_distances(X, Y, metric="cosine", n_jobs=2)
    >>> D_xy.shape
    (100, 60)
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"'X' must be 2-D, got shape {X.shape}")

    is_symmetric = Y is None
    if is_symmetric:
        Y = X
    else:
        Y = np.asarray(Y)
        if Y.ndim != 2:
            raise ValueError(f"'Y' must be 2-D, got shape {Y.shape}")
        if X.shape[1] != Y.shape[1]:
            raise ValueError(
                f"'X' and 'Y' must have the same number of features; "
                f"got X.shape[1]={X.shape[1]} and Y.shape[1]={Y.shape[1]}"
            )

    if chunk_size is not None and chunk_size < 1:
        raise ValueError(f"'chunk_size' must be >= 1, got {chunk_size}")

    n_workers = effective_n_jobs(n_jobs)

    # Single-worker fast path — skip joblib overhead entirely.
    if n_workers == 1:
        distance_matrix = cdist(X, Y, metric=metric, **metric_kwargs)
    
    # Multi-worker path with joblib parallelism.
    else:

        # Determine joblib backend.
        if backend is None:
            backend = (
                "threading"
                if isinstance(metric, str) and metric.lower() in _GIL_FREE_METRICS
                else "loky"
            )

        # Split X into row chunks, one per worker.
        # Cap n_chunks to n_samples so we never produce empty chunks.
        if chunk_size is None:
            n_chunks = min(n_workers, X.shape[0])
        else:
            n_chunks = min(
                max(1, int(np.ceil(X.shape[0] / chunk_size))),
                X.shape[0],
            )

        x_chunks = np.array_split(X, n_chunks)

        chunks_result: list[np.ndarray] = Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(_compute_chunk)(chunk, Y, metric, metric_kwargs)
            for chunk in x_chunks
        )

        distance_matrix = np.vstack(chunks_result)

    # Symmetrise if requested and if the result should be symmetric.
    if force_symmetry and is_symmetric:
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
        np.fill_diagonal(distance_matrix, 0) # ensure exact zeros on the diagonal after symmetrisation

    return distance_matrix
