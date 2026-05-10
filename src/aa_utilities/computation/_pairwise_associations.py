import itertools

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# a more advanced version compared to `scipy.stats.false_discovery_control`
from statsmodels.stats.multitest import multipletests

from ..storage import Container as C

def _infer_dtypes(df, enforce: dict=None):
    """
    Infer continuous (numeric) and string columns by dtype.
    Columns in `enforce` are forced to the specified type regardless of dtype.
    """
    enforce = enforce or {} # default to empty dict if None

    dtypes = pd.Series([''] * len(df.columns), index=df.columns)
    for col in df.columns:
        if col in enforce:
            dtypes.at[col] = enforce[col]
        elif pd.api.types.is_numeric_dtype(df[col]):
            dtypes.at[col] = 'numeric'
        elif pd.api.types.is_bool_dtype(df[col]): # treat boolean as categorical, tests via chi2
            dtypes.at[col] = 'string'
        else:
            dtypes.at[col] = 'string'
    return dtypes

def pairwise_associations(df, enforced_dtypes=None):
    """
    Run pairwise association tests for every unique pair of variables.

    Returns a DataFrame with columns:
        col_a, col_b, type_a, type_b, test,
        statistic, effect_size, n_samples, p_value, p_adj
    """

    # preparations
    dtypes = _infer_dtypes(df, enforce=enforced_dtypes)
    # creating a local copy to avoid modifying the original 
    # dataframe with fillna operations below
    df = df.copy()
    
    results = []
    for col_a, col_b in itertools.combinations(df.columns, 2):
        
        # determine variable types for the current pair
        a_is_cat = dtypes[col_a] == 'string'
        b_is_cat = dtypes[col_b] == 'string'

        # convert NaN to a category for categorical variables (if there is any NaN), 
        # but keep NaN for numeric variables.
        if a_is_cat and df[col_a].isna().any():
            df[col_a] = df[col_a].fillna('NaN')
        if b_is_cat and df[col_b].isna().any():
            df[col_b] = df[col_b].fillna('NaN')

        # only consider pairs with at least 5 complete samples
        pair = df[[col_a, col_b]].dropna(how='any')
        n_samples = len(pair)
        if n_samples < 5:
            continue

        # prepare a result object 
        result = C(
            col_a=col_a,
            col_b=col_b,
            type_a=dtypes[col_a],
            type_b=dtypes[col_b],
            test=np.nan,
            statistic=np.nan,
            effect_size=np.nan,
            p_value=np.nan,
            n_samples=n_samples,
        )

        # ── Continuous × Continuous: Spearman ρ ──────────────────────────
        # range: -1 (perfect negative) to 1 (perfect positive), 0 means no monotonic association
        if not a_is_cat and not b_is_cat:
            result.update(test='spearman')
            # if either variable is constant, correlation is undefined
            if pair[col_a].std() > 0 and pair[col_b].std() > 0:
                res = scipy_stats.spearmanr(pair[col_a], pair[col_b])
                result.update(
                    statistic=res.statistic,
                    effect_size=res.statistic,
                    p_value=res.pvalue,
                )
        
        # ── Categorical × Categorical: χ² + Cramér's V ───────────────────
        # range: 0 (no association) to 1 (perfect association)
        elif a_is_cat and b_is_cat:
            ct = pd.crosstab(pair[col_a], pair[col_b])
            r, c_dim = ct.shape
            # skip 1×k (or k×1) tables: one variable is constant after subsetting,
            # making the test uninformative (analogous to the std==0 skip for Spearman)
            if min(r, c_dim) < 2:
                continue
            # corrects for small sample sizes with Yates' correction (only for 2x2 tables)
            # This is conservative and inflates p-values for larger samples.
            chi2, p_value, _, _ = scipy_stats.chi2_contingency(ct, correction=True)
            v = np.sqrt(chi2 / (n_samples * (min(r, c_dim) - 1)))
            result.update(
                test='chi2',
                statistic=chi2,
                effect_size=np.clip(v, 0.0, 1.0),
                p_value=p_value,
            )

        # ── Continuous × Categorical: Kruskal-Wallis + ε ────────────────
        # range: ε²: 0 (no association) to 1 (perfect association), 
        # ε = √ε² is on correlation-like scale [0, 1]
        # note: correlation ratio η is another option which could be more comparable to 
        # pearson r (as visualized in a heatmap)
        else:
            assert a_is_cat != b_is_cat, "Exactly one variable must be categorical for this test."
            cat_col = col_a if a_is_cat else col_b
            cont_col = col_b if a_is_cat else col_a

            # skip if the continuous variable is constant (all ties → tie-correction denominator = 0)
            if pair[cont_col].std() == 0:
                continue

            # generate list of groups
            groups = [
                pair.loc[pair[cat_col] == g, cont_col].values
                for g in pair[cat_col].unique()
            ]

            # filter out empty groups
            groups = [g for g in groups if len(g) > 0]
            if len(groups) < 2:
                continue

            # perform Kruskal-Wallis test and calculate ε effect size
            H, p_value = scipy_stats.kruskal(*groups)
            k = len(groups)
            # epsilon-squared (variance-explained scale, [0, 1])
            eps2 = max(0.0, (H - k + 1) / (n_samples - k)) if n_samples > k else np.nan
            # epsilon = √ε² converts to correlation-like scale [0, 1],
            # comparable to Spearman |ρ| and Cramér's V
            # this scaling is somewhat arbitrary (and not a standard convention)
            eps = np.sqrt(eps2) if not np.isnan(eps2) else np.nan
            result.update(
                test='kruskal_wallis', 
                statistic=H, 
                effect_size=eps, 
                p_value=p_value,
            )

        # collecting the result for this pair
        results.append(result)

    # results concatenation and multiple testing correction
    results_df = pd.DataFrame(results)
    if len(results_df) > 0:
        _, p_adj, _, _ = multipletests(results_df['p_value'].fillna(1.0), method='fdr_bh')
        results_df['p_adj'] = p_adj
    else:
        results_df['p_adj'] = np.nan

    return results_df

