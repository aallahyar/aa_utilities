import importlib.metadata

import numpy as np
import pandas as pd
import pytest

from aa_utilities.computation.modeling._linear_models import _prepare_categoricals

def _installed_version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return 'not installed'


VERSIONS = f'rpy2={_installed_version("rpy2")}, pandas={pd.__version__}'
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
def model(rspace, skip_unless_ci):
    from aa_utilities.computation.modeling import LinearModel

    try:
        return LinearModel(space=rspace)
    except RuntimeError as exc:
        skip_unless_ci(f'R packages needed by LinearModel are unavailable: {exc}')


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


# ----- lineage: downstream steps refuse results whose upstream R objects changed -----


@pytest.fixture
def fitted(model, trial_df):
    model.set_data(trial_df)
    model.set_reference({'group': 'Low'})
    model.fit_lm('y ~ group')
    return model


def test_model_creates_its_own_rspace_when_none_is_given(skip_unless_ci):
    from aa_utilities.computation.modeling import LinearModel

    try:
        model = LinearModel()
    except (ImportError, OSError, RuntimeError) as exc:
        skip_unless_ci(f'R packages needed by LinearModel are unavailable: {exc}')
    assert model.R is not None


def test_fit_without_data_is_refused(model):
    from aa_utilities.computation.modeling import StaleResultError

    with pytest.raises(StaleResultError, match='set_data'):
        model.fit_lm('y ~ group')


def test_lineage_lists_a_token_per_valid_stage(fitted):
    assert set(fitted.lineage) == {'data', 'fit'}
    fitted.add_emmeans('group')
    fitted.add_contrasts()
    assert set(fitted.lineage) == {'data', 'fit', 'emmeans', 'contrasts'}


def test_refitting_replaces_the_fit_token_and_drops_downstream_stages(fitted):
    fit_token = fitted.lineage['fit']
    fitted.add_emmeans('group')
    fitted.fit_lm('y ~ group')
    assert fitted.lineage['fit'] != fit_token
    assert set(fitted.lineage) == {'data', 'fit'}
    assert 'ls_means' not in fitted.results


def test_new_data_invalidates_the_fit(fitted, trial_df):
    from aa_utilities.computation.modeling import StaleResultError

    fitted.set_data(trial_df)
    assert 'fit_coefs' not in fitted.results
    with pytest.raises(StaleResultError, match='fit'):
        fitted.add_emmeans('group')


def test_changing_the_reference_invalidates_the_fit(fitted):
    from aa_utilities.computation.modeling import StaleResultError

    fitted.set_reference({'group': 'High'})
    with pytest.raises(StaleResultError):
        fitted.add_emmeans('group')


def test_failed_refit_does_not_leave_the_previous_fit_usable(fitted):
    from aa_utilities.computation.modeling import StaleResultError

    with pytest.raises(RuntimeError):
        fitted.fit_lm('y ~ not_a_column')
    with pytest.raises(StaleResultError):
        fitted.add_emmeans('group')


def test_two_models_sharing_an_r_session_do_not_use_each_others_objects(fitted, trial_df):
    from aa_utilities.computation.modeling import LinearModel, StaleResultError

    other = LinearModel(space=fitted.R)
    other.set_data(trial_df)
    other.fit_lm('y ~ group')

    with pytest.raises(StaleResultError, match='another LinearModel'):
        fitted.add_emmeans('group')
    other.add_emmeans('group')  # the model that fitted last can still use it


def test_reassigning_the_fit_in_r_is_detected(fitted):
    from aa_utilities.computation.modeling import StaleResultError

    fitted.R('fit <- lm(y ~ 1, data = data)')
    with pytest.raises(StaleResultError, match='outside of this LinearModel'):
        fitted.add_emmeans('group')


def test_editing_the_data_in_r_after_the_fit_is_detected(fitted):
    from aa_utilities.computation.modeling import StaleResultError

    fitted.R('data <- data[1:30, ]')
    with pytest.raises(StaleResultError, match='data'):
        fitted.add_emmeans('group')


def test_editing_the_data_in_r_before_fitting_is_adopted_with_a_warning(model, trial_df, package_log):
    model.set_data(trial_df)
    model.R('data <- data[1:30, ]')
    model.fit_lm('y ~ group')
    assert model.results['n_observations'] == 30
    assert any('modified' in record.getMessage() for record in package_log.records)


def test_results_from_another_r_session_are_refused(fitted):
    from aa_utilities.computation.modeling import StaleResultError

    fitted.R('rm(list = "fit", envir = get(".__aa_token", envir = globalenv()))')  # what a fresh R session looks like
    with pytest.raises(StaleResultError, match='not created by this LinearModel'):
        fitted.add_emmeans('group')


def test_contrasts_can_be_appended_for_the_same_marginal_means(fitted):
    fitted.add_emmeans('group')
    fitted.add_contrasts(method='revpairwise')
    n_first = len(fitted.results.contrasts)
    fitted.add_contrasts(method='pairwise', append=True)
    assert len(fitted.results.contrasts) == 2 * n_first


@pytest.mark.parametrize('ci', [0, 1, 1.5, -0.1])
def test_confidence_level_must_be_strictly_between_0_and_1(model, trial_df, ci):
    model.set_data(trial_df)
    with pytest.raises(ValueError, match='ci'):
        model.fit_lm('y ~ group', ci=ci)


def test_confidence_level_is_not_rounded(model, trial_df):
    model.set_data(trial_df)
    model.fit_lm('y ~ group', ci=0.95)
    width_95 = (model.results.fit_coefs['conf.high'] - model.results.fit_coefs['conf.low']).iloc[0]
    model.fit_lm('y ~ group', ci=0.954)
    width_954 = (model.results.fit_coefs['conf.high'] - model.results.fit_coefs['conf.low']).iloc[0]
    assert width_954 > width_95


def test_repeated_contrast_labels_keep_their_own_context(model):
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            'group': np.tile(['A', 'B'], 40),
            'visit': np.repeat(['W1', 'W2'], 40),
            'y': rng.normal(size=80),
        }
    )
    model.set_data(df)
    model.fit_lm('y ~ group * visit')
    model.add_emmeans('group | visit')
    model.add_contrasts()

    contrasts = model.results.contrasts
    assert len(contrasts) == 2
    assert contrasts['visit_left'].tolist() == contrasts['visit_right'].tolist()
    assert sorted(contrasts['visit_left']) == ['W1', 'W2']
    assert model.get_contrast('group', ('B', 'A'), context={'visit': 'W2'}) is not None


def test_clearing_one_model_does_not_unregister_the_objects_of_another(fitted, trial_df):
    from aa_utilities.computation.modeling import LinearModel

    other = LinearModel(space=fitted.R)
    other.set_data(trial_df)
    other.fit_lm('y ~ group')

    fitted.clear_results(fit=True)  # the first model cleans up after itself

    other.add_emmeans('group')  # still the owner of `data` and `fit` in R


def test_data_removed_in_r_is_refused_instead_of_resolving_to_r_functions(fitted):
    from aa_utilities.computation.modeling import StaleResultError

    fitted.R('rm(data)')  # `data` would now resolve to the function utils::data
    with pytest.raises(StaleResultError, match='no longer exists'):
        fitted.set_reference({'group': 'High'})
    with pytest.raises(StaleResultError, match='no longer exists'):
        fitted.fit_lm('y ~ group')


def test_factorize_requires_data_set_by_this_model(model):
    from aa_utilities.computation.modeling import StaleResultError

    with pytest.raises(StaleResultError, match='set_data'):
        model.factorize()


def test_failed_set_reference_invalidates_the_fit_when_data_was_partly_changed(model):
    from aa_utilities.computation.modeling import StaleResultError

    model.set_data(pd.DataFrame({'y': [1.0, 2.0, 3.0, 4.0], 'g1': list('abab'), 'g2': list('xyxy')}))
    model.fit_lm('y ~ g1 + g2')

    with pytest.raises(RuntimeError):
        model.set_reference({'g1': 'b', 'g2': 'not_a_level'})  # g1 is re-leveled before g2 fails

    assert list(model.R('levels(data$g1)')) == ['b', 'a']
    assert 'fit' not in model.lineage
    with pytest.raises(StaleResultError):
        model.add_emmeans('g1')


def test_failed_set_reference_without_changes_keeps_the_fit(model):
    model.set_data(pd.DataFrame({'y': [1.0, 2.0, 3.0, 4.0], 'g1': list('abab')}))
    model.fit_lm('y ~ g1')

    with pytest.raises(RuntimeError):
        model.set_reference({'g1': 'not_a_level'})

    assert 'fit' in model.lineage


def test_invalid_confidence_level_keeps_the_previous_logistic_fit(model):
    model.set_data(pd.DataFrame({'y': [0, 1, 0, 1, 1, 0], 'g': list('abcabc')}))
    model.fit_logistic('y ~ g')
    with pytest.raises(ValueError, match='ci'):
        model.fit_logistic('y ~ g', ci=5)
    assert 'fit_coefs' in model.results and 'fit' in model.lineage


@pytest.fixture
def two_factor_df():
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            'group': np.tile(['A', 'B'], 20),
            'visit': np.repeat(['Week 1', 'Week 2'], 20),
            'num': np.repeat([1, 2], 20),
            'y': rng.normal(size=40),
        }
    )


def test_operator_of_contrasts_conditioned_on_another_predictor(model, two_factor_df):
    model.set_data(two_factor_df)
    model.fit_lm('y ~ group * visit')
    model.add_emmeans('group | visit')
    model.add_contrasts()
    assert model.results.contrasts['operator'].tolist() == ['-', '-']


def test_operator_of_crossed_contrasts_with_multi_word_levels(model, two_factor_df):
    model.set_data(two_factor_df)
    model.fit_lm('y ~ group * visit')
    model.add_emmeans('group:visit')
    model.add_contrasts()
    assert set(model.results.contrasts['operator']) == {'-'}


def test_numeric_predictors_keep_their_values_in_contrasts(model, two_factor_df):
    model.set_data(two_factor_df)
    model.fit_lm('y ~ group * factor(num)')
    model.add_emmeans('num')
    model.add_contrasts()

    contrasts = model.results.contrasts
    assert contrasts['operator'].tolist() == ['-']
    assert (contrasts['num_left'].iloc[0], contrasts['num_right'].iloc[0]) == (2, 1)
    assert model.get_contrast('num', (2, 1)) is not None
