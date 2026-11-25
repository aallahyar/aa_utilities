import numpy as np
import pandas as pd

from aa_utilities.computation import pairwise_associations


def test_categorical_column_with_missing_values_does_not_crash():
    df = pd.DataFrame(
        {
            'a': pd.Categorical(['a', None, 'b', 'a', 'b', 'a']),
            'b': [1, 2, 3, 4, 5, 6.0],
            'c': list('xyxyxy'),
        }
    )
    result = pairwise_associations(df)
    assert len(result) == 3


def test_boolean_columns_are_tested_as_categorical():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            'flag_a': rng.random(40) > 0.5,
            'flag_b': rng.random(40) > 0.5,
            'value': rng.normal(size=40),
        }
    )
    result = pairwise_associations(df).set_index(['col_a', 'col_b'])

    assert result.loc[('flag_a', 'flag_b'), 'test'] == 'chi2'
    assert result.loc[('flag_a', 'value'), 'test'] == 'kruskal_wallis'
