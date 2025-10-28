import numpy as np
import pandas as pd

from ...storage import Container

class LinearModel:

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

    @classmethod # the function does not need the instantiated object
    def get_dummy(n=500, n_visit=5, seed=42):
        # prepare a dummy data
        rng = np.random.default_rng(seed=seed)
        n_subj = n // n_visit
        
        dummy_df = (
            pd.DataFrame()
            .assign(
                idx=range(n),
                SUBJID=lambda df: df.idx.map(lambda i: f'S{i % n_subj:04}'),
                USUBJID=np.repeat([f'S{si:04}' for si in range(n_subj)], n_visit),
                TRT01P=lambda df: np.where(df.USUBJID.str[-1].astype(int) % 2 == 0, 'Placebo', 'Treatment'),
                # TRT01P=lambda df: np.where(df.idx % 2 == 0, 'Placebo', 'Treatment'),
                VISIT_idx=np.tile(range(n_visit), n // n_visit),
                AVISIT=lambda df: df.VISIT_idx.map(lambda vi: f'Week {vi}'),
                BASE=rng.normal(loc=1000, scale=500, size=n).astype(int),
                # BASE=lambda df: rng.lognormal(5, 0.25, size=len(df)),
                AVAL=lambda df: df.BASE - df.BASE * (df.VISIT_idx / 10) * np.where(df.TRT01P == 'Placebo', 0.5, 1),
                # AVAL=lambda df: np.where(df.TRT01P == 'Placebo', rng.lognormal(df.VISIT_idx, 0.3), rng.lognormal(df.VISIT_idx + 0.0, 0.3)),
                CHANGE=lambda df: df.AVAL - df.BASE,
            )
            .assign(
                BASE=lambda df: df.groupby('USUBJID').BASE.transform(lambda g: g.iat[0]),
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
    def set_data(self, df, remove_categories=True, factorize=True):

        # data adjustments
        df = df.copy() # make a local copy
        if remove_categories:
            for col in df.select_dtypes(include='category').columns:
                df[col] = df[col].astype(str)
        self.R['data'] = df.copy()
        if factorize:
            self.factorize()

        self.results['n_samples'] = len(df)

    # @staticmethod # used when no other methods/variables of the Class are needed
    def factorize(self, columns: list[str] = None):
        if columns is None: # Factorize all columns of type object (string)
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
                    # print(c(factor_name, 'is found'))
                    ref <- factor_references[[factor_name]]
                    if (is.factor(data[[factor_name]])) {
                        data[[factor_name]] <- relevel(data[[factor_name]], ref = ref)
                    } else {
                        data[[factor_name]] <- relevel(factor(data[[factor_name]], ordered = FALSE), ref = ref)
                    }
                }
            }
        """)

    def get_model_formula(self):
        self.R(f"""
            # chatgpt: deparse is more reliable than capture.output(print(...))
            # model_formula <- capture.output(print(formula(fit)))
            model_formula <- deparse(formula(fit))
        """)
        if isinstance(self.R['model_formula'], (str, )):
            formula = self.R['model_formula']
        else:
            formula = ' '.join(line.strip() for line in self.R['model_formula'])
        return formula

    def fit_lm(self, formula, ci=0.95):
        # e.g., formula = 'TRT01P'
        # optional: family=gaussian(link = "identity") or gaussian(link = "log"
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
        self.results['model_name'] = 'lm'
        self.results['formula'] = self.get_model_formula()
        self.results['n_observations'] = int(self.R['n_observations'])
        self.results['fit_coefs'] = self.R['fit_coefs'].set_index('term')

    def fit_logistic(self, formula, ci=0.95):
        if ci is None:
            broom_params = f'conf.int = FALSE'
        else:
            broom_params = f'conf.int = TRUE, conf.level = {ci:0.2f}'
            
        self.R(f"""
            fit <- glm(
                formula = {formula}, 
                family = binomial(link = "logit"), 
                data = data
            )
        """)

        # collect result
        self.R(f"""
            n_observations <- nobs(fit)
            fit_coefs <- broom::tidy(fit, {broom_params})
        """)
        self.results['model_name'] = 'logistic'
        self.results['formula'] = self.get_model_formula()
        self.results['n_observations'] = int(self.R['n_observations'])
        self.results['fit_coefs'] = self.R['fit_coefs'].set_index('term')

    def fit_mmrm(self, formula, ci=0.95):
        # Having BASE on the right-hand side: Considers that higher/lower baseline values may have a different effect on Response.
        # formula = 'Response ~ BASE + TRT01P + AVISIT + TRT01P:AVISIT + us(AVISIT | USUBJID) + confounders'
        self.R(f"""
        fit <- mmrm::mmrm(
            formula = {formula},
            data = data,
            method = "Kenward-Roger"
        )
        """)

        # collect result
        self.R(f"""
            n_observations <- mmrm::component(fit)[['n_obs']]
            n_subjects <- mmrm::component(fit)[['n_subjects']]
        """)

        # Extract coefficients, statistics and confidence intervals at specified level
        self.R(f"""
        coef_df <- as.data.frame(summary(fit)$coefficients)
        conf_df <- as.data.frame(confint(fit, level = {ci:0.2f}))
        """)
        assert self.R['coef_df'].index.equals(self.R['conf_df'].index), "Mismatch in coefficient indices between coef_df and conf_df."
        assert self.R['conf_df'].shape[1] == 2, "conf_df should have exactly two columns for confidence intervals."
        fit_coefs = (
            self.R['coef_df']
            .assign(**{
                'conf.low': self.R['conf_df'].iloc[:, 0],
                'conf.high': self.R['conf_df'].iloc[:, 1],
            })
            .rename(columns={
                'Estimate': 'estimate',
                'Std. Error': 'std.error',
                't value': 'statistic',
                'Pr(>|t|)': 'p.value',
            })
            .rename_axis(index='term')
            [['estimate', 'std.error', 'df', 'conf.low', 'conf.high', 'statistic', 'p.value']]
        )

        self.results['model_name'] = 'mmrm'
        self.results['formula'] = self.get_model_formula()
        self.results['n_observations'] = int(self.R['n_observations'])
        self.results['n_subjects'] = int(self.R['n_subjects'])
        self.results['fit_coefs'] = fit_coefs

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
        
        self.R(f"""
            # type = "response" , # Estimates are back-transformed to the response scale (e.g., probabilities if you fit a logistic model).
            # type = "link" ,  # Estimates are shown on the linear predictor scale. For example, you’d see logits for logistic regression.
            # exponentiate=TRUE # for logistic regression, exponentiates the log-odds to odds ratios.
            LSmeans <- emmeans::emmeans(fit, spec = ~ {spec}, type="{scale}", level = {ci:0.2f}{emm_kws})
            LSmeans_td <- broom::tidy(LSmeans, conf.int = TRUE, conf.level = {ci:0.2f})
            # print(LSmeans_td)
            
            LSmeans_attrs <- attributes(LSmeans)
            predictors <- LSmeans_attrs$roles$predictors
        """)
        if isinstance(self.R['predictors'], str):
            predictors = [self.R['predictors']]
        else:
            predictors = self.R['predictors'].tolist()
        self.results['ls_means'] = self.R['LSmeans_td'].set_index(predictors)

    def add_contrasts(self, method='revpairwise', ci=0.95, append=False):
        # method: "revpairwise", "pairwise", "eff", "del.eff"
        # eff: compare each level with the average over all
        # del.eff: compare each level with average over all other levels
        # comparisons are done in the original response scale (defined in formula), irrespective of emmeans(type=X). 
        # If emmeans(type='response') are called beforehand, the contrasts are exponentiated, and so are named A / B (but the p-values stay the same).
        self.R(f"""
            # `pairs()` is a special case of `contrast()`
            # emm_diff <- pairs(LSmeans, adjust = "none", reverse = TRUE)
            emm_diff <- emmeans::contrast(LSmeans, method="{method}", adjust = "none")
            emm_diff_td <- broom::tidy(emm_diff, conf.int = TRUE, conf.level = {ci:0.2f})
            # print(emm_diff_td, width = Inf, n = Inf)
        """)

        if append:
            if 'contrasts' not in self.results: # initialize an empty DataFrame, if it does not exist
                self.results['contrasts'] = pd.DataFrame()
            
            self.results['contrasts'] = pd.concat([
                self.results['contrasts'],
                self.R['emm_diff_td'].set_index('contrast'),
            ], axis=0, ignore_index=False)
        else:
            self.results['contrasts'] = self.R['emm_diff_td'].set_index('contrast')
        
        # extracting details per Arm and Timepoint
        # self.R['pw_diff_td'].contrast.str.extract(
        #     r'^(?P<Arm>Teze 210 mg Q4W|Placebo)' # named groups
        #     r'.*(Week \d+)'
        #     r' - (Teze 210 mg Q4W|Placebo)'
        #     r'.*(Week \d+)'
        # )


