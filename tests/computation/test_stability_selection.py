import numpy as np
from sklearn.linear_model import Lasso

from aa_utilities.computation.modeling import StabilitySelectionConfig, StabilitySelector
from aa_utilities.computation.modeling.stability_selection._stability_selection import SelectorResult
from aa_utilities.computation.modeling.stability_selection._subsampling import StandardSampler


def _data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 5))
    y = X[:, 0] + rng.normal(scale=0.1, size=40)
    return X, y


def _selector(estimator, feature_names):
    return SelectorResult(output=np.abs(estimator.coef_) > 0)


def _run(selector_fn=_selector, estimator_factory=lambda seed=None: Lasso(alpha=0.1)):
    X, y = _data()
    selector = StabilitySelector(
        estimator_factory=estimator_factory,
        selector_fn=selector_fn,
        subsampler=StandardSampler(ratio=0.5),
        config=StabilitySelectionConfig(n_iterations=3),
    )
    return selector.fit(X, y)


def test_successful_fits_have_no_failure():
    run = _run()
    assert len(run.results) == 3
    assert all(result.failure is None and result.output is not None for result in run.results)


def test_failure_reported_by_the_selector_is_preserved():
    def failing_selector(estimator, feature_names):
        return SelectorResult(output=None, failure='did not converge')

    run = _run(selector_fn=failing_selector)
    assert [result.failure for result in run.results] == ['did not converge'] * 3


def test_failing_estimator_factory_is_recorded_per_fit():
    def broken_factory(seed=None):
        raise ValueError('cannot build the estimator')

    run = _run(estimator_factory=broken_factory)
    assert len(run.results) == 3
    assert all('cannot build the estimator' in result.failure for result in run.results)
