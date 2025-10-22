class LinearModel:

    def __init__(self, space=None):
        self.R = space
        
        self.R("""
            library(tidyverse)
            # # library(mmrm)
            # # library(MASS) # Modern Applied Statistics with S
            
            # sets the width of output terminal window
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
        if columns is None:
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
        if isinstance(self.R['model_formula'], (np.ndarray, pd.Series)):
            formula = ' '.join(line.strip() for line in self.R['model_formula'])
        else:
            formula = self.R['model_formula']
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
            fit_coefs <- broom::tidy(fit, conf.int = TRUE, conf.level = {ci:0.2f})
        """)
        self.results['model_name'] = 'mmrm'
        self.results['formula'] = self.get_model_formula()
        self.results['n_observations'] = int(self.R['n_observations'])
        self.results['n_subjects'] = int(self.R['n_subjects'])
        self.results['fit_coefs'] = self.R['fit_coefs'].set_index('term')

    def add_emmeans(self, spec, scale='link', ci=0.95, emm_kws=''):
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
            # type = "response" , type = "link" , 
            # exponentiate=TRUE
            LSmeans <- emmeans::emmeans(fit, spec = ~ {spec}, type="{scale}", level = {ci:0.2f}, rg.limit = 100000{emm_kws})
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

    def examples():
        raise NotImplementedError('This part is only for history keeping. It was not meant to be executed!')
        
        # pyright: ignore[reportUnreachable]
        # absolute change from baseline, response scale = linear
        rng = np.random.default_rng(seed=42)
        n = 1000
        n_visit = 5
        ci = 0.95
        data = (
            pd.DataFrame()
            .assign(
                USUBJID=np.repeat([f'S{si}' for si in range(n // 5)], 5),
                TRT01P=lambda df: np.where(df.USUBJID.str[-1].astype(int) % 2 == 0, 'Placebo', 'Treatment'),
                VISIT_idx=np.tile(range(n_visit), reps=n // n_visit),
                AVISIT=lambda df: df.VISIT_idx.map(lambda vi: f'Week {vi}'),
                BASE=rng.normal(loc=100, scale=25, size=n).astype(int),
            )
            .assign(
                BASE=lambda df: df.groupby('USUBJID').BASE.transform(lambda g: g.iat[0]),
                AVAL=lambda df: df.BASE + df.VISIT_idx * np.where(df.TRT01P == 'Placebo', 1, 3),
                CHG=lambda df: df.AVAL - df.BASE,
            )
        )
        
        model = LinearModel(space=R)
        model.set_data(data, remove_categories=True, factorize=True)
        model.fit_lm(formula=f'CHG ~ TRT01P * AVISIT', ci=ci)
        model.add_emmeans(spec=f'TRT01P * AVISIT', scale='response', ci=ci)
        print(model.results.ls_means[['estimate']].round(2))
        
        #                   estimate
        # TRT01P    AVISIT          
        # Placebo   Week 0      -0.0
        # Treatment Week 0       0.0
        # Placebo   Week 1       1.0
        # Treatment Week 1       3.0
        # Placebo   Week 2       2.0
        # Treatment Week 2       6.0
        # Placebo   Week 3       3.0
        # Treatment Week 3       9.0
        # Placebo   Week 4       4.0
        # Treatment Week 4      12.0

        # percentage change from baseline, response scale = linear        
        # same as "absolute change from baseline", with the following modifications:
        AVAL=lambda df: df.BASE + df.BASE * df.VISIT_idx / 10 * np.where(df.TRT01P == 'Placebo', 1, 3)
        PRC_CHG=lambda df: (df.AVAL - df.BASE) * 100 / df.BASE
        model.fit_lm(formula=f'PRC_CHG ~ TRT01P * AVISIT', ci=ci)
        
        #                   estimate
        # TRT01P    AVISIT          
        # Placebo   Week 0      -0.0
        # Treatment Week 0      -0.0
        # Placebo   Week 1      10.0
        # Treatment Week 1      30.0
        # Placebo   Week 2      20.0
        # Treatment Week 2      60.0
        # Placebo   Week 3      30.0
        # Treatment Week 3      90.0
        # Placebo   Week 4      40.0
        # Treatment Week 4     120.0

        # percentage change from baseline, response scale = log-normal
        pseudocount = 1e-20
        BASE=rng.lognormal(mean=0, sigma=1, size=n)
        AVAL=lambda df: df.BASE + df.BASE * df.VISIT
        LOG_CHG=lambda df: np.log(df.AVAL + pseudocount) - np.log(df.BASE + pseudocount)
        model.fit_lm(formula=f'LOG_CHG ~ TRT01P * AVISIT', ci=ci)
        model.add_emmeans(spec=f'TRT01P * AVISIT', scale='response', ci=ci)
        print(np.exp(model.results.ls_means[['estimate']]).sub(1).mul(100).round(2))
        
        #                   estimate
        # TRT01P    AVISIT          
        # Placebo   Week 0      -0.0
        # Treatment Week 0      -0.0
        # Placebo   Week 1      10.0
        # Treatment Week 1      30.0
        # Placebo   Week 2      20.0
        # Treatment Week 2      60.0
        # Placebo   Week 3      30.0
        # Treatment Week 3      90.0
        # Placebo   Week 4      40.0
        # Treatment Week 4     120.0

        # absolute change from baseline, response scale = log-normal
        pseudocount = 1e-20
        BASE=np.exp(rng.normal(loc=0, scale=1, size=n)) # or rng.lognormal(mean, sigma)
        AVAL=lambda df: df.BASE + df.VISIT_idx * np.where(df.TRT01P == 'Placebo', 10, 30)
        LOG_AVAL=lambda df: np.log(df.AVAL + pseudocount)
        
        model.fit_lm(formula=f'LOG_AVAL ~ TRT01P * AVISIT', ci=ci)
        model.add_emmeans(spec=f'TRT01P * AVISIT', scale='link', ci=ci)
        print(np.exp(model.results.ls_means[['estimate']]).sub(pseudocount).round(2))
        
        #                   estimate
        # TRT01P    AVISIT          
        # Placebo   Week 0      0.98
        # Treatment Week 0      1.00
        # Placebo   Week 1     11.38
        # Treatment Week 1     31.76
        # Placebo   Week 2     21.66
        # Treatment Week 2     61.58
        # Placebo   Week 3     31.48
        # Treatment Week 3     91.52
        # Placebo   Week 4     41.31
        # Treatment Week 4    121.56

        # absolute change from baseline, response scale = log-normal, using MMRM
        model.fit_mmrm(formula=f'LOG_AVAL ~ TRT01P * AVISIT + us(AVISIT | USUBJID)', ci=ci)
        
        #                   estimate
        # TRT01P    AVISIT          
        # Placebo   Week 0      0.98
        # Treatment Week 0      1.00
        # Placebo   Week 1     11.38
        # Treatment Week 1     31.76
        # Placebo   Week 2     21.66
        # Treatment Week 2     61.58
        # Placebo   Week 3     31.48
        # Treatment Week 3     91.52
        # Placebo   Week 4     41.31
        # Treatment Week 4    121.56
