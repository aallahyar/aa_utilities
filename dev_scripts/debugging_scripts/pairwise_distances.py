"""
Compare aa_utilities.computation.pairwise_distances against
scipy.spatial.distance.cdist across several scenarios.

Run from the repo root:
    python dev_scripts/debugging_scripts/pairwise_distances.py
"""

import timeit

import numpy as np
from scipy.spatial.distance import cdist

from aa_utilities.computation import pairwise_distances

rng = np.random.default_rng(seed=42)

# ── helpers ──────────────────────────────────────────────────────────────────

def _header(title: str) -> None:
    width = 60
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")


def _check_close(D_ours: np.ndarray, D_ref: np.ndarray, label: str) -> None:
    max_diff = np.max(np.abs(D_ours - D_ref))
    match = "✓" if np.allclose(D_ours, D_ref, rtol=1e-9) else "✗"
    print(f"  {match}  max|ours − scipy| = {max_diff:.2e}   [{label}]")


# ── 1. Basic all-vs-all (euclidean) ──────────────────────────────────────────

_header("1. All-vs-all, euclidean")

X = rng.standard_normal((200, 50))

D_scipy = cdist(X, X, metric="euclidean")
D_ours  = pairwise_distances(X, metric="euclidean")

print(f"  output shape : {D_ours.shape}")
print(f"  diagonal (should be 0): {np.diag(D_ours)[:5]}")
_check_close(D_ours, D_scipy, "euclidean all-vs-all")


# ── 2. X-vs-Y (cosine) ───────────────────────────────────────────────────────

_header("2. X vs Y, cosine")

Y = rng.standard_normal((80, 50))

D_scipy = cdist(X, Y, metric="cosine")
D_ours  = pairwise_distances(X, Y, metric="cosine")

print(f"  output shape : {D_ours.shape}")
_check_close(D_ours, D_scipy, "cosine X-vs-Y")


# ── 3. force_symmetry ────────────────────────────────────────────────────────

_header("3. force_symmetry=True (correlation metric)")

D_plain = pairwise_distances(X, metric="correlation")
D_sym   = pairwise_distances(X, metric="correlation", force_symmetry=True)

asymmetry_plain = np.max(np.abs(D_plain - D_plain.T))
asymmetry_sym   = np.max(np.abs(D_sym   - D_sym.T))

print(f"  max asymmetry without force_symmetry : {asymmetry_plain:.2e}")
print(f"  max asymmetry with    force_symmetry : {asymmetry_sym:.2e}")


# ── 4. Custom callable metric ─────────────────────────────────────────────────

_header("4. Custom callable metric (weighted L1)")

weights = rng.uniform(0.5, 2.0, size=X.shape[1])

def weighted_l1(u: np.ndarray, v: np.ndarray) -> float:
    return float(np.sum(weights * np.abs(u - v)))

D_scipy = cdist(X, X, metric=weighted_l1)
D_ours  = pairwise_distances(X, metric=weighted_l1)

_check_close(D_ours, D_scipy, "custom callable")


# ── 5. metric_kwargs forwarding (Minkowski p=3) ───────────────────────────────

_header("5. metric_kwargs forwarding (Minkowski p=3)")

D_scipy = cdist(X, X, metric="minkowski", p=3)
D_ours  = pairwise_distances(X, metric="minkowski", p=3)

_check_close(D_ours, D_scipy, "minkowski p=3")


# ── 6. Multi-core timing comparison ──────────────────────────────────────────

_header("6. Multi-core timing (n=2000, p=200, euclidean)")

X_large = rng.standard_normal((2000, 200))
repeats = 3

t_scipy = timeit.timeit(
    lambda: cdist(X_large, X_large, metric="euclidean"),
    number=repeats,
) / repeats

t_single = timeit.timeit(
    lambda: pairwise_distances(X_large, metric="euclidean", n_jobs=1),
    number=repeats,
) / repeats

t_multi = timeit.timeit(
    lambda: pairwise_distances(X_large, metric="euclidean", n_jobs=4),
    number=repeats,
) / repeats

print(f"  scipy cdist (1 core)      : {t_scipy:.3f}s")
print(f"  pairwise_distances n_jobs=1: {t_single:.3f}s")
print(f"  pairwise_distances n_jobs=4: {t_multi:.3f}s")
print(f"  speedup vs single-core     : {t_single / t_multi:.2f}x")


# ── 7. Backend auto-selection ─────────────────────────────────────────────────

_header("7. Backend auto-selection (n_jobs=2)")

X_small = rng.standard_normal((100, 10))

# String metric → threading
import aa_utilities.computation._distance as _dist
from unittest.mock import patch

captured_backends: list[str] = []

original_parallel = _dist.Parallel

class _SpyParallel(original_parallel):
    def __init__(self, **kwargs):
        captured_backends.append(kwargs.get("backend", "?"))
        super().__init__(**kwargs)

with patch.object(_dist, "Parallel", _SpyParallel):
    pairwise_distances(X_small, metric="euclidean", n_jobs=2)
    pairwise_distances(X_small, metric=lambda u, v: float(np.sum(np.abs(u - v))), n_jobs=2)

print(f"  string metric  → backend selected: '{captured_backends[0]}'")
print(f"  callable metric → backend selected: '{captured_backends[1]}'")

print()
