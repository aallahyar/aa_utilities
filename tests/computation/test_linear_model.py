import importlib.metadata

import numpy as np
import pandas as pd
import pytest

from aa_utilities.computation.modeling._linear_models import _prepare_categoricals

VERSIONS = f'rpy2={importlib.metadata.version("rpy2")}, pandas={pd.__version__}'
DRIFT_HINT = f'The rpy2/pandas conversion boundary may have changed ({VERSIONS}); re-verify downstream results.'


def _r_list(rspace, expression):
    rspace(f'probe_ <- {expression}')
    value = rspace['probe_']  # RSpace turns length-1 vectors into scalars and longer ones into Series
    return list(value) if isinstance(value, pd.Series) else [value]


# ----- _prepare_categoricals (pure pandas) -----


def test_prepare_drops_unused_categories_and_keeps_order():
    df = pd.DataFrame({'g': pd.Categorical(['a', 'c', 'a'], categories=['c', 'b', 'a', 'z'])})
    out, _ = _prepare_categoricals(df)
    assert list(out['g'].cat.categories) == ['c', 'a']


def test_prepare_keeps_missing_values():
    df = pd.DataFrame({'g': pd.Categorical(['a', None, 'b'], categories=['b', 'a'])})
    out, _ = _prepare_categoricals(df)
    assert out['g'].isna().tolist() == [False, True, False]


def test_prepare_does_not_mutate_input():
    df = pd.DataFrame({'g': pd.Categorical(['a'], categories=['a', 'z'])})
    _prepare_categoricals(df)
    assert list(df['g'].cat.categories) == ['a', 'z']


def test_prepare_leaves_other_columns_untouched():
    df = pd.DataFrame({'s': ['x', 'y'], 'n': [1.0, 2.0]})
    out, ordered = _prepare_categoricals(df)
    pd.testing.assert_frame_equal(out, df)
    assert ordered == []


def test_prepare_reports_ordered_columns_only():
    df = pd.DataFrame(
        {
            'o': pd.Categorical(['a', 'b'], categories=['a', 'b'], ordered=True),
            'u': pd.Categorical(['a', 'b'], categories=['a', 'b'], ordered=False),
        }
    )
    _, ordered = _prepare_categoricals(df)
    assert ordered == ['o']


# ----- rpy2 / pandas conversion tripwires -----


def test_unordered_categorical_becomes_plain_factor(rspace):
    rspace['data'] = pd.DataFrame({'col': pd.Categorical(['a', 'b'], categories=['a', 'b', 'c'])})
    assert _r_list(rspace, 'class(data$col)') == ['factor'], DRIFT_HINT
    assert _r_list(rspace, 'levels(data$col)') == ['a', 'b', 'c'], DRIFT_HINT
    assert _r_list(rspace, 'as.character(is.ordered(data$col))') == ['FALSE'], DRIFT_HINT


def test_ordered_categorical_becomes_ordered_factor_with_polynomial_contrasts(rspace):
    values = ['a', 'b', 'c'] * 2
    rspace['data'] = pd.DataFrame(
        {
            'y': [1.0, 2.5, 2.0, 1.5, 3.0, 2.2],
            'col': pd.Categorical(values, categories=['a', 'b', 'c'], ordered=True),
        }
    )
    assert _r_list(rspace, 'class(data$col)') == ['ordered', 'factor'], DRIFT_HINT
    rspace('fit <- lm(y ~ col, data = data)')
    assert _r_list(rspace, 'names(coef(fit))') == ['(Intercept)', 'col.L', 'col.Q'], DRIFT_HINT


@pytest.mark.parametrize(
    'make_series',
    [
        lambda: pd.Categorical(['a', None, 'b', 'a'], categories=['b', 'a', 'z']),
        lambda: pd.Series(['a', None, 'b', 'a'], dtype=object),
        lambda: pd.Series(['a', np.nan, 'b', 'a'], dtype='str'),
    ],
    ids=['categorical', 'object', 'str'],
)
def test_missing_values_reach_r_as_na(rspace, make_series):
    rspace['data'] = pd.DataFrame({'col': make_series()})
    assert _r_list(rspace, 'is.na(data$col)') == [False, True, False, False], DRIFT_HINT


# ----- LinearModel (requires R with tidyverse, broom, emmeans, mmrm) -----


@pytest.fixture
def model(rspace):
    from aa_utilities.computation.modeling import LinearModel

    try:
        return LinearModel(space=rspace)
    except RuntimeError as exc:
        pytest.skip(f'R packages needed by LinearModel are unavailable: {exc}')


@pytest.fixture
def trial_df():
    rng = np.random.default_rng(0)
    group = np.repeat(['Low', 'Mid', 'High'], 20)
    effect = pd.Series({'Low': 0.0, 'Mid': 1.0, 'High': 2.5})
    return pd.DataFrame({'y': effect[group].to_numpy() + rng.normal(size=len(group)), 'group': group})


def _fit_and_summarize(model, df):
    model.set_data(df)
    model.set_reference({'group': 'Low'})
    model.fit_lm('y ~ group')
    model.add_emmeans('group')
    model.add_contrasts()
    return model.results.ls_means.sort_index(), model.results.contrasts.sort_index()


def test_set_data_keeps_declared_order_and_drops_unused_levels(model, trial_df):
    trial_df['group'] = pd.Categorical(trial_df['group'], categories=['Mid', 'Placebo', 'Low', 'High', 'Unused'])
    model.set_data(trial_df)
    assert _r_list(model.R, 'levels(data$group)') == ['Mid', 'Low', 'High']


def test_factorize_does_not_reorder_existing_factors(model, trial_df):
    trial_df['group'] = pd.Categorical(trial_df['group'], categories=['Mid', 'Low', 'High'])
    trial_df['site'] = ['B', 'A'] * (len(trial_df) // 2)
    model.set_data(trial_df)
    assert _r_list(model.R, 'levels(data$group)') == ['Mid', 'Low', 'High']
    assert _r_list(model.R, 'levels(data$site)') == ['A', 'B']


def test_set_reference_moves_only_the_named_level_to_front(model):
    model.set_data(pd.DataFrame({'col': pd.Categorical(list('abcd') * 2, categories=list('abcd'))}))
    model.set_reference({'col': 'c'})
    assert _r_list(model.R, 'levels(data$col)') == ['c', 'a', 'b', 'd'], DRIFT_HINT


def test_set_reference_rejects_ordered_factor(model):
    model.set_data(pd.DataFrame({'col': pd.Categorical(['a', 'b'], categories=['a', 'b'], ordered=True)}))
    with pytest.raises(RuntimeError, match='ordered factor'):
        model.set_reference({'col': 'b'})


def test_set_data_warns_about_ordered_categoricals(model, package_log):
    df = pd.DataFrame({'col': pd.Categorical(['a', 'b'], categories=['a', 'b'], ordered=True)})
    model.set_data(df)
    assert any('Ordered categorical' in record.getMessage() for record in package_log.records)


def test_string_and_categorical_inputs_give_identical_results(model, trial_df):
    ls_str, contrasts_str = _fit_and_summarize(model, trial_df)

    categorical_df = trial_df.assign(
        group=pd.Categorical(trial_df['group'], categories=['Unused', 'High', 'Mid', 'Low', 'Other'])
    )
    ls_cat, contrasts_cat = _fit_and_summarize(model, categorical_df)

    pd.testing.assert_frame_equal(ls_str, ls_cat)
    pd.testing.assert_frame_equal(contrasts_str, contrasts_cat)


def test_ordered_categorical_leaves_marginal_means_unchanged(model, trial_df):
    categories = ['Low', 'Mid', 'High']

    def fit_means(ordered):
        df = trial_df.assign(group=pd.Categorical(trial_df['group'], categories=categories, ordered=ordered))
        model.set_data(df)
        model.fit_lm('y ~ group')
        model.add_emmeans('group')
        return model.results.ls_means.sort_index()

    pd.testing.assert_frame_equal(fit_means(ordered=False), fit_means(ordered=True))


def test_negative_binomial_with_offset_yields_contrasts(model):
    rng = np.random.default_rng(0)
    group = np.repeat(['Low', 'High'], 60)
    exposure = rng.uniform(1, 2, size=len(group))
    counts = rng.poisson(exposure * np.where(group == 'High', 5.0, 3.0))
    model.set_data(pd.DataFrame({'count': counts, 'exposure': exposure, 'group': group}))
    model.set_reference({'group': 'Low'})

    model.fit_negbin('count ~ offset(log(exposure)) + group')
    model.add_emmeans('group')
    model.add_contrasts()

    contrasts = model.results.contrasts
    assert len(contrasts) == 1
    assert pd.api.types.is_float_dtype(contrasts['null.value'])  # broom returns it as a matrix column for offset models
