"""
Concurrency tests for RSpace.

Two examples:
  1. loky backend (multiprocessing) — safe: each worker has its own R interpreter.
  2. threading backend              — CRASHES: rpy2 conversion rules are stored in a
     Python contextvars.ContextVar. Worker threads inherit a copy of the context as it
     existed at thread-creation time, which has no active conversion context. As a result
     rpy2's get_conversion() raises NotImplementedError inside every worker thread —
     the code crashes before any logical race even has a chance to manifest.

Run with:
    python concurrency_tests.py
"""

import time

from joblib import Parallel, delayed

# ---------------------------------------------------------------------------
# Worker helpers
# ---------------------------------------------------------------------------

def _loky_worker(value: float) -> float:
    """Instantiated inside the worker process — gets a private R interpreter."""
    from aa_utilities.wrappers import RSpace  # import inside worker to avoid pickling issues

    R = RSpace()
    R['x'] = value
    result = R('Sys.sleep(0.05); x^2')  # small sleep to force overlapping execution
    return result


def _threading_worker(R, value: float, results: list, idx: int) -> None:
    """Uses a shared RSpace instance — demonstrates the race condition."""
    R['x'] = value                      # writes to the shared R global env
    time.sleep(0.01)                    # yield so another thread can overwrite x
    result = R('x^2')                   # may read a different thread's x
    results[idx] = result


# ---------------------------------------------------------------------------
# Example 1: loky (multiprocessing) — SAFE
# ---------------------------------------------------------------------------

def example_loky(inputs: list):
    print("\n=== Example 1: loky backend (safe) ===")
    expected = [x**2 for x in inputs]

    results = Parallel(n_jobs=4, backend='loky')(
        delayed(_loky_worker)(x) for x in inputs
    )

    print(f"  inputs:   {inputs}")
    print(f"  expected: {expected}")
    print(f"  got:      {results}")
    print(f"  match:    {results == expected}")


# ---------------------------------------------------------------------------
# Example 2: threading — CRASHES (ContextVar not propagated to worker threads)
# ---------------------------------------------------------------------------

def example_threading(inputs: list):
    print("\n=== Example 2: threading backend (crashes — ContextVar not propagated) ===")
    print("  rpy2 stores conversion rules in a contextvars.ContextVar.")
    print("  Worker threads inherit the context at creation time, which has no active")
    print("  conversion context → rpy2 raises NotImplementedError in every thread.")
    print("  Note: this crashes before any logical race condition can even manifest.")

    from aa_utilities.wrappers import RSpace  # single shared instance

    R = RSpace()
    results = [None] * len(inputs)

    try:
        Parallel(n_jobs=4, backend='threading')(
            delayed(_threading_worker)(R, x, results, i)
            for i, x in enumerate(inputs)
        )
    except RuntimeError as exc:
        print(f"\n  ✗ Crashed with: {type(exc).__name__}: {exc}")
        print("  → Use the loky (multiprocessing) backend instead.")


# ---------------------------------------------------------------------------

if __name__ == '__main__':
    inputs = [1.0, 2.0, 3.0, 4.0]
    example_loky(inputs)
    example_threading(inputs)
