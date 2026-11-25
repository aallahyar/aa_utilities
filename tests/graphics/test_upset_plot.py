import matplotlib

matplotlib.use('Agg')

import pytest

from aa_utilities.graphics import UpsetPlot


def test_set_sizes_ignore_duplicated_elements():
    upset = UpsetPlot({'A': [1, 1, 2], 'B': [2, 3]})
    assert upset.sets.loc['A', 'size'] == 2
    assert upset.intersections.loc['A', '_size'] == 2


def test_generators_are_accepted():
    upset = UpsetPlot({'A': (i for i in range(3)), 'B': (i for i in range(2, 5))})
    assert upset.sets['size'].tolist() == [3, 3]
    assert upset.intersections.loc['A & B', '_size'] == 1


@pytest.mark.parametrize('name', ['_size', '_n_included', '_n_excluded'])
def test_names_of_metadata_columns_are_rejected(name):
    with pytest.raises(ValueError, match='reserved'):
        UpsetPlot({name: [1], 'B': [2]})


def test_empty_sets_are_rejected():
    with pytest.raises(ValueError):
        UpsetPlot({})


def test_filter_keeps_distinct_patterns_whose_labels_collide():
    upset = UpsetPlot({'A': [1, 2], 'B': [2, 3], 'A & B': [9]})
    upset.filter(lambda ix: ix)
    patterns = upset.intersections[['A', 'B', 'A & B']].drop_duplicates()
    assert len(patterns) == len(upset.intersections) == 7
