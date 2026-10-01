import numpy as np
import pandas as pd

from ...storage import Container


class LinearModel:
    _ONE_VS_ONE_METHODS = {'pairwise', 'revpairwise'}  # contrast methods that always compare a single level vs. a single level

    def __init__(self, space=None):
        self.R = space

        self.R("""
            library(tidyverse)
            
            # Not needed as we refer to functions with ::
            # library(broom)
            # library(emmeans)
            # library(mmrm)
            # library(MASS) # Modern Applied Statistics with S
            
            # sets the width of the output terminal window
            options(width=180)

            # check package availability
            if (!requireNamespace("broom", quietly=TRUE)) stop("Package 'broom' is not installed.")
            if (!requireNamespace("emmeans", quietly=TRUE)) stop("Package 'emmeans' is not installed.")
            if (!requireNamespace("mmrm", quietly=TRUE)) stop("Package 'mmrm' is not installed.")
        """)

        self.results = Container(
            # is_factored=False,
        )
        self.results._pp.display_width = 175

    @staticmethod  # the function does not need the instantiated object
    def get_dummy(n=500, n_visit=5, seed=42):
        # prepare a dummy data
        rng = np.random.default_rng(seed=seed)
        n_subj = n // n_visit

        dummy_df = (
            pd.DataFrame()
            .assign(
                idx=range(n),
                # SUBJID=lambda df: df.idx.map(lambda i: f'S{i % n_subj:04}'),
                SUBJID=np.repeat(range(n_subj), n_visit),
                USUBJID=np.repeat([f'S{si:04}' for si in range(n_subj)], n_visit),
                TRT01P=lambda df: np.where(df.USUBJID.str[-1].astype(int) % 2 == 0, 'Placebo', 'Treatment'),
                # TRT01P=lambda df: np.where(df.idx % 2 == 0, 'Placebo', 'Treatment'),
                VISIT_idx=np.tile(range(n_visit), n // n_visit),
                AVISIT=lambda df: df.VISIT_idx.map(lambda vi: f'Week {vi}'),

                # BASE=rng.normal(loc=1000, scale=500, size=n).astype(int),
                # BASE=lambda df: rng.lognormal(5, 0.25, size=len(df)),
                BASE=lambda df: df.groupby('USUBJID')['idx'].transform(lambda g: rng.normal(loc=1000, scale=500)).astype(int), # one random baseline draw per subject, broadcast across all of that subject's visits
                AVAL=lambda df: df.BASE - df.BASE * (df.VISIT_idx / 10) * np.where(df.TRT01P == 'Placebo', 0.5, 1),
                # AVAL=lambda df: np.where(df.TRT01P == 'Placebo', rng.lognormal(df.VISIT_idx, 0.3), rng.lognormal(df.VISIT_idx + 0.0, 0.3)),
                CHANGE=lambda df: df.AVAL - df.BASE,
            )
            .assign(
                # BASE=lambda df: df.groupby('USUBJID').BASE.transform(lambda g: g.iat[0]),
                # AVAL=lambda df: np.where(df.AVISIT == 'V0', df.BASE, df.BASE + df.visit_idx + rng.uniform(0, 0.05, size=n)),
                # AVAL=lambda df: np.where(df.AVISIT == 'V0', df.BASE, df.BASE * np.exp(1) + rng.uniform(0, 10.1, size=n)),
                # AVAL=lambda df: np.where(df.AVISIT == 'V0', df.BASE, df.BASE + df.visit_idx + rng.uniform(0, 0.05, size=n)),
                # CHG=lambda df: df.AVAL - df.BASE,
                # log_change=lambda df: np.log(df.AVAL) - np.log(df.BASE),
            )
            .assign(
                # AVAL=lambda df: df.AVAL + np.where((df.TRT01P == 'Placebo') | df.AVISIT == 'V0', 0, 1),
            )
        )
        return dummy_df

    # @classmethod # used when other methods/variables of the Class are needed
    def set_data(self, df, remove_categories=True, preserve_na=True, factorize=True):

        # data adjustments
        df = df.copy()  # make a local copy
        if remove_categories:
            for col in df.select_dtypes(include='category').columns:
                is_na = df[col].isna()
                df[col] = df[col].astype(str)
                if preserve_na:
                    df.loc[is_na, col] = np.nan
        self.R['data'] = df.copy()
        if factorize:
            self.factorize()

        self.results['n_samples'] = len(df)

    # @staticmethod # used when no other methods/variables of the Class are needed
    def factorize(self, columns: list[str] = None):
        if columns is None:  # Factorize all columns of type object (string)
            self.R("""
                data <- data %>% mutate(across(where(is.character), as.factor))
            """)
        else:
            self.R['to_factor'] = self.R.ro.StrVector(columns)
            self.R("""
                data <- data %>% mutate(across(to_factor, as.factor))
            """)

    def set_reference(self, references: dict):
        # example: {'TRT01P': 'Placebo', 'AVISIT': 'Week 0'}
        self.R['factor_references'] = self.R.ro.ListVector(references)
        self.R("""
            for (factor_name in names(factor_references)) {
                if (factor_name %in% colnames(data)){
                    ref <- factor_references[[factor_name]]
                    if (is.factor(data[[factor_name]])) {
                        data[[factor_name]] <- relevel(data[[factor_name]], ref = ref)
                    } else {
                        data[[factor_name]] <- relevel(factor(data[[factor_name]], ordered = FALSE), ref = ref)
                    }
                }
            }
        """)

    def clear_results(self, fit=False, emmeans=False, contrasts=False):
        if fit:
            self.results.pop('fit_coefs', None)
            self.results.pop('n_observations', None)
            self.results.pop('formula', None)
            self.results.pop('model_name', None)
            self.results.pop('warnings', None)
            self.clear_results(emmeans=True)
        if emmeans:
            self.results.pop('ls_means', None)
            self.results.pop('predictors', None)
            self.clear_results(contrasts=True)
        if contrasts:
            self.results.pop('contrasts', None)

    def get_model_formula(self):
        self.R("""
            # chatgpt: deparse is more reliable than capture.output(print(...))
            # model_formula <- capture.output(print(formula(fit)))
            model_formula <- deparse(formula(fit))
        """)
        if isinstance(self.R['model_formula'], (str,)):
            formula = self.R['model_formula']
        else:
            formula = ' '.join(line.strip() for line in self.R['model_formula'])
        return formula

    def fit_lm(self, formula, ci=0.95):
        self.clear_results(fit=True)

        with self.R.capture_warnings() as warnings:
            # e.g., formula = 'TRT01P'
            self.R(f"""
                fit <- lm(
                    formula = {formula},
                    data = data,
                )
            """)

            # collect result
            self.R(f"""
            fit_coefs <- broom::tidy(fit, conf.int = TRUE, conf.level = {ci:0.2f})
            n_observations <- nobs(fit)
            """)
        self.results.setdefault('warnings', []).extend(warnings)
        self.results['model_name'] = 'lm'
        self.results['formula'] = self.get_model_formula()
        self.results['n_observations'] = int(self.R['n_observations'])
        self.results['fit_coefs'] = self.R['fit_coefs'].set_index('term')

    def fit_logistic(self, formula, ci=0.95):
        self.clear_results(fit=True)
        if ci is None:
            broom_params = 'conf.int = FALSE'
        else:
            broom_params = f'conf.int = TRUE, conf.level = {ci:0.2f}'

        with self.R.capture_warnings() as warnings:
            self.R(f"""
                fit <- glm(
                    formula = {formula}, 
                    # other options: family=gaussian(link = "identity") or gaussian(link = "log")
                    family = binomial(link = "logit"), 
                    data = data
                )
            """)

            # collect result
            self.R(f"""
                n_observations <- nobs(fit)
                fit_coefs <- broom::tidy(fit, {broom_params})
            """)
        self.results.setdefault('warnings', []).extend(warnings)
        self.results['model_name'] = 'logistic'
        self.results['formula'] = self.get_model_formula()
        self.results['n_observations'] = int(self.R['n_observations'])
        self.results['fit_coefs'] = self.R['fit_coefs'].set_index('term')

    def fit_mmrm(self, formula, ci=0.95):
        """
        Having BASE on the right-hand side: Considers that higher/lower baseline values may have a different effect on Response.
        formula = 'Response ~ BASE + TRT01P + AVISIT + TRT01P:AVISIT + us(AVISIT | USUBJID) + confounders'
        """
        self.clear_results(fit=True)
        with self.R.capture_warnings() as warnings:
            self.R(f"""
            fit <- mmrm::mmrm(
                formula = {formula},
                data = data,
                method = "Kenward-Roger"
            )
            """)

            # collect result
            self.R("""
                n_observations <- mmrm::component(fit)[['n_obs']]
                n_subjects <- mmrm::component(fit)[['n_subjects']]
            """)

            # Extract coefficients, statistics and confidence intervals at specified level
            self.R(f"""
            coef_df <- as.data.frame(summary(fit)$coefficients)
            conf_df <- as.data.frame(confint(fit, level = {ci:0.2f}))
            """)
        self.results.setdefault('warnings', []).extend(warnings)
        assert self.R['coef_df'].index.equals(self.R['conf_df'].index), (
            'Mismatch in coefficient indices between coef_df and conf_df.'
        )
        assert self.R['conf_df'].shape[1] == 2, 'conf_df should have exactly two columns for confidence intervals.'
        fit_coefs = (
            self.R['coef_df']
            .assign(
                **{
                    'conf.low': self.R['conf_df'].iloc[:, 0],
                    'conf.high': self.R['conf_df'].iloc[:, 1],
                }
            )
            .rename(
                columns={
                    'Estimate': 'estimate',
                    'Std. Error': 'std.error',
                    't value': 'statistic',
                    'Pr(>|t|)': 'p.value',
                }
            )
            .rename_axis(index='term')[['estimate', 'std.error', 'df', 'conf.low', 'conf.high', 'statistic', 'p.value']]
        )

        self.results['model_name'] = 'mmrm'
        self.results['formula'] = self.get_model_formula()
        self.results['n_observations'] = int(self.R['n_observations'])
        self.results['n_subjects'] = int(self.R['n_subjects'])
        self.results['fit_coefs'] = fit_coefs

    def fit_negbin(self, formula, exponentiate=True, ci=0.95):
        """Fits a negative binomial regression model using the MASS::glm.nb function in R.
        The mean-variance relation is Var(Y) = μ + μ²/θ, so larger θ means less overdispersion.

        Parameters:
            formula (str): The model formula as a string.
                E.g., ``'EXACN ~ offset(log(TMEXRISK)) + TRT01P'``.
            exponentiate (bool): Whether to exponentiate the coefficients (to get incidence rate ratios).
            ci (float): Confidence interval level (e.g., 0.95 for 95% CI).
        """
        self.clear_results(fit=True)
        self.R['exponentiate'] = exponentiate

        with self.R.capture_warnings() as warnings:
            self.R(f"""
                fit <- MASS::glm.nb(
                    formula = {formula},
                    data = data,
                    link = log
                )
            """)

            # collect result
            self.R(f"""
            summary_fit <- summary(fit)
            fit_theta <- summary_fit$theta
            # fit_theta_se  <- summary_fit$SE.theta
            fit_coefs <- broom::tidy(fit, conf.int = TRUE, conf.level = {ci:0.2f}, exponentiate = exponentiate)
            n_observations <- nobs(fit)
            """)
        self.results.setdefault('warnings', []).extend(warnings)
        self.results['model_name'] = 'negative_binomial'
        self.results['formula'] = self.get_model_formula()
        self.results['n_observations'] = int(self.R['n_observations'])
        self.results['fit_theta'] = float(self.R['fit_theta'])
        self.results['fit_coefs'] = self.R['fit_coefs'].set_index('term')

    def add_emmeans(self, spec, scale='link', ci=0.95, emm_kws=', rg.limit = 100000'):
        # add estimated marginal means (EMMs), or Least-squares means to `self.results`
        # lm: spec = 'TRT01P'
        # mmrm: spec = 'TRT01P:AVISIT'

        # """ example for changing reference grid to response scale before calculating the contrasts
        # model <- lm(log(conc) ~ source + factor(percent), data = emmeans::pigs)
        # ls.means <- emmeans::emmeans(model, spec = ~ source, type='response')
        # print(ls.means)
        #  source response   SE df lower.CL upper.CL
        #  fish       29.8 1.09 23     27.6     32.1
        #  soy        39.1 1.47 23     36.2     42.3
        #  skim       44.6 1.75 23     41.1     48.3

        # ls.means <- emmeans::regrid(ls.means, transform = "response")
        # pw_diff <- emmeans::contrast(ls.means, method="revpairwise", type='link', adjust='none')
        # print(pw_diff)
        #  contrast    estimate   SE df t.ratio p.value
        #  soy - fish      9.34 1.85 23   5.063  <.0001
        #  skim - fish    14.76 2.08 23   7.098  <.0001
        #  skim - soy      5.41 2.23 23   2.424  0.0236
        # """

        # clean up old `results`, if any
        self.clear_results(emmeans=True)

        with self.R.capture_warnings() as warnings:
            self.R(f"""
                # type:
                #   * "response": # Estimates are back-transformed to the response scale (e.g., probabilities if you fit a logistic model). Note that only back-transforms when it can detect the outcome came from a recognized link/transform (e.g. `log(y) ~`) written directly in the model formula, or a GLM family/link). Otherwise, it works as `link`
                #   * "link" :    # Estimates are shown on the linear predictor scale. For example, you see logits for logistic regression.
                LSmeans <- emmeans::emmeans(fit, spec = ~ {spec}, type="{scale}", level = {ci:0.2f}{emm_kws})
                LSmeans_td <- broom::tidy(LSmeans, conf.int = TRUE, conf.level = {ci:0.2f})
                # print(LSmeans_td)
                
                LSmeans_attrs <- attributes(LSmeans)
                predictors <- LSmeans_attrs$roles$predictors
            """)
        self.results.setdefault('warnings', []).extend(warnings)
        if isinstance(self.R['predictors'], str):
            predictors = [self.R['predictors']]
        else:
            predictors = self.R['predictors'].tolist()
        self.results['predictors'] = predictors
        self.results['ls_means'] = self.R['LSmeans_td'].set_index(predictors)

    def add_contrasts(self, method='revpairwise', ci=0.95, append=False):
        """
        method: "revpairwise", "pairwise", "eff", "del.eff"
        eff: compare each level with the average over all
        del.eff: compare each level with average over all other levels
        Contrasts operate on the scale of the emmeans object's reference grid.
        If emmeans(type='response') was called, contrasts are on the response scale
        (e.g., ratios for log-link models, named A / B). P-values remain unchanged.

        For one-vs-one contrast methods (`pairwise`, `revpairwise`), the single `contrast`
        label index is additionally split into per-side predictor columns (named
        `{predictor}_left`/`{predictor}_right`, per the emmeans spec order), plus `label`
        (the raw contrast string) and `operator` columns -- this enables `get_contrast()`.
        Other methods (e.g. `eff`/`del.eff`, which compare a level against a combination of
        several others) keep only their raw label, since they aren't a clean one-vs-one split.
        """
        if not append:
            self.clear_results(contrasts=True)
        
        with self.R.capture_warnings() as warnings:
            self.R(f"""
                # `pairs()` is a special case of `contrast()`
                # emm_diff <- pairs(LSmeans, adjust = "none", reverse = TRUE)
                emm_diff <- emmeans::contrast(LSmeans, method="{method}", adjust = "none")
                emm_diff_td <- broom::tidy(emm_diff, conf.int = TRUE, conf.level = {ci:0.2f})
                # print(emm_diff_td, width = Inf, n = Inf)
            """)

            contrasts_df = self.R['emm_diff_td'].set_index('contrast')
            if method in self._ONE_VS_ONE_METHODS:
                contrasts_df = self._expand_contrasts(contrasts_df)
        self.results.setdefault('warnings', []).extend(warnings)

        if append:
            if 'contrasts' not in self.results:  # initialize an empty DataFrame, if it does not exist
                self.results['contrasts'] = pd.DataFrame()

            self.results['contrasts'] = pd.concat(
                [
                    self.results['contrasts'],
                    contrasts_df,
                ],
                axis=0,
                ignore_index=False,
            )
        else:
            self.results['contrasts'] = contrasts_df

    def _expand_contrasts(self, contrasts_df):
        """Split each one-vs-one contrast label into per-side predictor levels.

        Uses the contrast object's own linear coefficients (`emmeans::coef(emm_diff)`)
        to identify, for each contrast, which reference-grid row(s) it compares
        (positive vs. negative coefficient), rather than parsing the label text.

        Only called for one-vs-one contrast methods (see `add_contrasts`), so every row
        is guaranteed to have exactly one +1 and one -1 coefficient.
        """

        # predictors are cached on `self.results` by `add_emmeans`
        predictors = self.results.predictors

        # get the linear coefficients for each contrast from the emmeans object
        self.R("""
            # one column per contrast (in the same order as emm_diff_td$contrast) holding
            # the linear coefficient applied to each reference-grid row (`c.1`, `c.2`, ...)
            # (coef.emmGrid is an S3 method registered by emmeans but not exported,
            # so it must be called via the base generic rather than emmeans::coef)
            contrast_coefs <- as.data.frame(coef(emm_diff), check.names = FALSE)
        """)
        grid = (
            self.R['contrast_coefs']
            .set_index(predictors)
            .T
        )
        assert len(grid) == len(contrasts_df), (
            'Mismatch between number of contrasts in coef(emm_diff) and emm_diff_td.'
        )

        # get individual predictor levels for each side of the contrast
        comb1 = pd.DataFrame(data='', columns=predictors, index=contrasts_df.index)
        comb2 = pd.DataFrame(data='', columns=predictors, index=contrasts_df.index)
        operators = pd.DataFrame(data='', columns=['operator'], index=contrasts_df.index)
        for label, (_, cnt) in zip(contrasts_df.index, grid.iterrows()):
            preds1 = cnt.loc[cnt == +1].index
            preds2 = cnt.loc[cnt == -1].index
            assert len(preds1) == 1 and len(preds2) == 1, "Each contrast must have exactly one +1 and one -1 coefficient"
            comb1.loc[label, :] = preds1[0]
            comb2.loc[label, :] = preds2[0]
            side1 = ' '.join(comb1.loc[label])
            side2 = ' '.join(comb2.loc[label])
            operators.loc[label, 'operator'] = label[len(side1):len(label) - len(side2)].strip()
        expanded = pd.concat([
            contrasts_df, 
            comb1.add_suffix('_left'), 
            operators, 
            comb2.add_suffix('_right'),
        ], axis=1)

        return expanded

        # extracting details per Arm and Timepoint
        # self.R['pw_diff_td'].contrast.str.extract(
        #     r'^(?P<Arm>Teze 210 mg Q4W|Placebo)' # named groups
        #     r'.*(Week \d+)'
        #     r' - (Teze 210 mg Q4W|Placebo)'
        #     r'.*(Week \d+)'
        # )

    def get_contrast(self, factor, levels, context=None):
        """Looks up a single pairwise contrast from `self.results.contrasts`.

        Args:
            factor (str): the predictor whose two levels are being compared (must be one
                of the predictors in the fitted `contrast_spec`, e.g. 'group').
            levels (tuple): `(level_new, level_ref)` -- the two levels of `factor` being
                compared, in the same order as the underlying contrast (matching
                `revpairwise`'s "new / old" or "new - old" convention).
            context (dict, optional): additional predictor -> value pairs that must be
                held fixed (equal on both sides of the contrast), e.g.
                `{'AVISIT': 'Week 24', 'arm_abbr': 'TzH'}`. Use `None` (default) or `{}`
                when there are no other contrasted factors (e.g. a single-factor model).

        Returns:
            pandas.Series: the single matching row of `self.results.contrasts` (with all
                of its columns, e.g. `estimate`, `conf.low`, `conf.high`, `statistic`,
                `p.value`), or `None` if that combination wasn't estimated (e.g. a sparse
                cell that doesn't appear in the data).

        Raises:
            ValueError: if `factor` or a `context` key isn't a predictor that was actually
                expanded into `self.results.contrasts` (e.g. a typo, or the contrasts were
                fit with a non-one-vs-one method such as `eff`/`del.eff`, which are never
                expanded -- see `add_contrasts`). Also raised if `factor` is also present
                in `context`, or if more than one row matches (indicating an ambiguous
                contrast specification).

        Example:
            # single-factor model (e.g. a cross-sectional comparison with no other
            # contrasted factors)
            model.get_contrast(factor='group', levels=('High', 'Low'))

            # multi-factor interaction model (e.g. arm_abbr * group * AVISIT), scoped to
            # one specific visit and one specific arm panel
            model.get_contrast(
                factor='group', levels=('ADO', 'ADL'),
                context={'AVISIT': 'Week 24', 'arm_abbr': 'TzH'},
            )
        """
        if context is None:
            context = {}
        if factor in context:
            raise ValueError(f'`factor` ({factor!r}) must not also appear in `context`.')

        contrasts = self.results.contrasts
        level_new, level_ref = levels

        # factor must differ between the two sides; every context entry must be equal
        # on both sides (i.e. "held fixed") -- both expressed the same way below so a
        # single loop can build the mask and validate column names uniformly.
        comparisons = {factor: (level_new, level_ref)} | {k: (v, v) for k, v in context.items()}

        mask = pd.Series(True, index=contrasts.index)
        for col, (val_left, val_right) in comparisons.items():
            if f'{col}_left' not in contrasts.columns:
                raise ValueError(
                    f'{col!r} is not a contrasted predictor. Available predictors: '
                    f'{self.results.predictors!r}. (Note: contrasts are only expanded for '
                    f'one-vs-one methods -- see add_contrasts().)'
                )
            mask &= (contrasts[f'{col}_left'] == val_left) & (contrasts[f'{col}_right'] == val_right)

        matches = contrasts.loc[mask]
        if len(matches) == 0:
            return None
        if len(matches) > 1:
            raise ValueError(f'Ambiguous contrast match for factor={factor!r}, levels={levels}, context={context}')
        return matches.iloc[0]

    def __repr__(self):
        out = 'LinearModel'
        meta = []
        if 'model_name' in self.results:
            meta.append(f'model_name={self.results["model_name"]}')
        if 'formula' in self.results:
            meta.append(f'formula={self.results["formula"]}')
        if len(meta) != 0:
            out += '(' + ', '.join(meta) + ')'
        return out
