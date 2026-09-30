"""
Demonstrates RSpace's warning capture.

`RSpace.warnings` is an append-only log — it is NOT reset before every `R(...)` call
(unlike older behavior), so warnings raised across several calls belonging to the same
logical operation (e.g. fit -> emmeans -> contrasts) are not lost.

Use `RSpace.capture_warnings()` to scope which warnings belong to a given block of
`R(...)` calls, regardless of how many calls that block makes.

Run with:
    python warnings_capture.py
"""

from aa_utilities.wrappers import RSpace


# ---------------------------------------------------------------------------
# Example 1: warnings accumulate across multiple R() calls
# ---------------------------------------------------------------------------

def example_accumulation():
    print("\n=== Example 1: warnings persist across calls (no per-call reset) ===")
    R = RSpace()

    R('warning("first call")', convert=False)
    R('warning("second call")', convert=False)

    print(f"  R.warnings has {len(R.warnings)} entr(y/ies):")
    for w in R.warnings:
        print(f"    {w!r}")


# ---------------------------------------------------------------------------
# Example 2: capture_warnings() scopes warnings to one logical operation
# ---------------------------------------------------------------------------

def example_capture_scope():
    print("\n=== Example 2: capture_warnings() scopes a block of calls ===")
    R = RSpace()

    R('warning("before the scope, not captured")', convert=False)

    with R.capture_warnings() as warnings:
        R('warning("inside scope, call 1")', convert=False)
        R('warning("inside scope, call 2")', convert=False)

    print(f"  captured (this operation only): {len(warnings)} entr(y/ies)")
    for w in warnings:
        print(f"    {w!r}")
    print(f"  R.warnings (full session log): {len(R.warnings)} entr(y/ies)")


# ---------------------------------------------------------------------------
# Example 3: mirrors how LinearModel attaches warnings to `self.results`
# ---------------------------------------------------------------------------

def example_results_attachment():
    print("\n=== Example 3: attaching captured warnings to a results dict ===")
    R = RSpace()
    results = {}

    # e.g. a "fit" step that issues a convergence warning
    with R.capture_warnings() as warnings:
        R('warning("model did not converge")', convert=False)
        R('x <- 1 + 1', convert=False)  # a second call in the same logical step
    results.setdefault('warnings', []).extend(warnings)

    # e.g. a subsequent "emmeans" step that issues its own warning
    with R.capture_warnings() as warnings:
        R('warning("some levels are not estimable")', convert=False)
    results.setdefault('warnings', []).extend(warnings)

    print(f"  results['warnings'] has {len(results['warnings'])} entr(y/ies):")
    for w in results['warnings']:
        print(f"    {w!r}")


if __name__ == '__main__':
    example_accumulation()
    example_capture_scope()
    example_results_attachment()
