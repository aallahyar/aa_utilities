import numpy as np
import pandas as pd
import pytest

from aa_utilities.computation.modeling import remove_effects


@pytest.fixture
def df():
    rng = np.random.default_rng(1)
    n = 60
    age = rng.normal(50, 10, n)
    return pd.DataFrame(
        {
            'y': 0.5 * age + rng.normal(size=n),
            'age': age,
            'age_baseline': rng.normal(size=n),
            'sex': rng.choice(['F', 'M'], n),
        }
    )


def test_removing_a_term_does_not_remove_terms_sharing_its_prefix(df):
    fit = remove_effects(df, 'y', covs_all=['age', 'age_baseline'], covs_remove=['age'])
    assert fit.covs_removed == ['age']
    assert fit.covs_kept == ['Intercept', 'age_baseline']


def test_keeping_a_term_does_not_keep_terms_sharing_its_prefix(df):
    fit = remove_effects(df, 'y', covs_all=['age', 'age_baseline'], covs_keep=['age'])
    assert fit.covs_kept == ['Intercept', 'age']
    assert fit.covs_removed == ['age_baseline']


def test_categorical_term_removes_all_of_its_levels(df):
    fit = remove_effects(df, 'y', covs_all=['age', 'sex'], covs_remove=['sex'])
    assert fit.covs_removed == ['sex[T.M]']


def test_empty_removal_list_removes_nothing(df):
    fit = remove_effects(df, 'y', covs_all=['age'], covs_remove=[])
    assert fit.covs_removed == []
    np.testing.assert_allclose(fit.response_adjusted, df['y'])


def test_empty_keep_list_keeps_only_the_intercept(df):
    fit = remove_effects(df, 'y', covs_all=['age'], covs_keep=[])
    assert fit.covs_kept == ['Intercept']


def test_keeping_terms_works_for_a_formula_without_an_intercept():
    frame = pd.DataFrame({'y': [2.0, 4.0, 6.0, 8.0], 'x': [1.0, 2.0, 3.0, 4.0]})
    fit = remove_effects(frame, 'y', covs_all=['0', 'x'], covs_keep=['x'])
    assert fit.covs_kept == ['x']
    np.testing.assert_allclose(fit.response_adjusted, frame['y'])
