import matplotlib

matplotlib.use('Agg')

import numpy as np
import pytest

from aa_utilities.graphics import jitter, to_colors


@pytest.mark.parametrize('make', [list, np.array], ids=['list', 'ndarray'])
def test_jitter_accepts_lists_and_arrays(make):
    values = make([1.0, 1.1, 1.2, 5.0, 9.0, 9.1])
    offsets = jitter(values)
    assert offsets.shape == (6,)
    assert np.isfinite(offsets).all()



def test_numeric_values_with_none_are_colored_as_numbers():
    with_none = to_colors([0, None, 1, 100])
    without = to_colors([0, 1, 100])
    assert with_none[1] == '#808080'
    assert [with_none[0], with_none[2], with_none[3]] == without


def test_missing_values_among_strings_get_the_default_color():
    colors = to_colors(['a', np.nan, 'b', None])
    assert colors[1] == colors[3] == '#808080'
    assert colors[0] != '#808080' and colors[2] != '#808080' and colors[0] != colors[2]


def test_strings_are_still_categorical():
    colors = to_colors(['x', 'y', 'x'])
    assert colors[0] == colors[2] != colors[1]


def test_two_dimensional_input_is_rejected_with_a_clear_error():
    with pytest.raises(ValueError, match='1D vector'):
        to_colors([[1, 2], [3, 4]])
