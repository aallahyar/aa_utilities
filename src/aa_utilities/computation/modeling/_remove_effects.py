import re

import numpy as np
import patsy
import statsmodels.api as sm

from ...loggers import setup_logger
from ..._configurations import configs

# setup logger
logger = setup_logger(name=__name__, level=configs.log.level)


def remove_effects(dataframe, response, covs_all, covs_remove=None, covs_keep=None, verbose=False):
    """
    Remove the effects of specified covariates from the response variable in a dataframe.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The input dataframe containing the response and covariates.
    response : str
        The name of the response variable.
    covs_all : list of str
        The list of all covariates to include in the model.
    covs_remove : list of str, optional
        The list of covariates to remove from the response. Either `covs_remove` or `covs_keep` must be provided.
    covs_keep : list of str, optional
        The list of covariates to keep in the response. Either `covs_remove` or `covs_keep` must be provided.
    verbose : bool, default False
        If True, debug information will be logged.

    Returns
    -------
    fit : statsmodels.regression.linear_model.RegressionResultsWrapper
        The fitted model with an additional attribute `response_adjusted` containing the adjusted response.
    """
    
    assert (covs_remove is None) ^ (covs_keep is None), 'Provide either `covs_remove` or `covs_keep`, not both.'
    if covs_remove is not None:
        assert np.isin(covs_remove, covs_all).all(), 'Some covariates in `covs_remove` are not present in `covs_all`.'
    if covs_keep is not None:
        assert np.isin(covs_keep, covs_all).all(), 'Some covariates in `covs_keep` are not present in `covs_all`.'

    # fitting the model
    formula = f'{response} ~ ' + ' + '.join(covs_all)
    y, X = patsy.dmatrices(
        formula, data=dataframe, return_type='dataframe', NA_action=patsy.NAAction(NA_types=[])
    )  # NaN values are considered as an independent category
    model = sm.OLS(y.iloc[:, 0], X)
    fit = model.fit()
    if verbose:
        logger.debug(f'formula: {formula}\nDesign matrix:\n{X.head()}\n{fit.summary()}')

    # verify column presence
    missing = set(X.columns) - set(fit.params.index)
    if len(missing) > 0:
        logger.warning(f'Missing columns in the fitted model (likely dropped due to collinearity): {sorted(missing)}')

    # determine which columns to keep
    if covs_remove is not None:
        # guard against an empty list: '|'.join([]) is '', which would otherwise match every column name
        if len(covs_remove) == 0:
            columns_drop = []
        else:
            columns_drop = (
                X
                # column names are suffixed with their levels by patsy. So `.*` is needed to match them.
                .filter(regex=r'^(' + '|'.join(map(re.escape, covs_remove)) + ').*')
                .columns
                .tolist()
            )
        if verbose:
            logger.debug(f'Columns to drop: {columns_drop}')
        columns_keep = X.drop(columns=columns_drop).columns.tolist()
    else:
        if len(covs_keep) == 0:
            columns_keep = X.filter(regex=r'^Intercept.*')
        else:
            columns_keep = X.filter(regex=r'^(Intercept|' + '|'.join(map(re.escape, covs_keep)) + ').*')
        columns_keep = columns_keep.columns.tolist()

    if verbose:
        logger.debug(f'Columns to keep: {columns_keep}')

    # adjusting response levels
    # assert X.columns.equals(fit.params.index) # we no longer require this, as we align params below
    assert X.index.equals(fit.resid.index), 'Sample index mismatch between design and residuals'
    # align params to design columns; missing coefficients (dropped due to collinearity) -> 0
    if verbose and not X.columns.equals(fit.params.index):
        logger.debug('Fitted parameters will be aligned to design matrix columns.')
    params_aligned = fit.params.reindex(X.columns).fillna(0)
    fit.response_adjusted = X[columns_keep].mul(params_aligned[columns_keep], axis=1).sum(axis=1).add(fit.resid)

    return fit
