import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from aa_utilities.graphics import LinkArtists, links

plt.switch_backend('Agg')


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')


@pytest.fixture
def ax():
    _, axes = plt.subplots()
    return axes


def _draw_boxes(ax, orient='vertical'):
    # category positions 0..2 on one axis, values up to 140 on the other (no boxplot: its orientation kwarg differs across matplotlib versions)
    categories, values = (ax.set_xlim, ax.set_ylim) if orient == 'vertical' else (ax.set_ylim, ax.set_xlim)
    categories(-0.5, 2.5)
    values(0, 140)
    return ax


# ----- both orientations -----


def test_vertical_bar_clears_the_given_bases(ax):
    _draw_boxes(ax)
    result = links([(0, 1)], ['p=0.01'], bases={0: 99, 1: 139}, ax=ax)

    link = result.links[0]
    assert link.bar_value > 139
    assert link.line.get_ydata()[0] == pytest.approx(99)  # left foot starts at its own base
    assert ax.get_ylim()[1] >= link.bar_value


def test_horizontal_bar_clears_the_given_bases(ax):
    _draw_boxes(ax, 'horizontal')
    result = links([(0, 1)], ['p=0.01'], bases={0: 99, 1: 139}, orient='horizontal', ax=ax)

    link = result.links[0]
    assert link.bar_value > 139
    assert link.line.get_xdata()[0] == pytest.approx(99)  # left foot starts at its own base
    assert ax.get_xlim()[1] >= link.bar_value
    assert list(link.line.get_ydata()[[0, -1]]) == [0, 1]  # feet sit at the linked y-positions


def test_horizontal_does_not_touch_ylim(ax):
    _draw_boxes(ax, 'horizontal')
    before = ax.get_ylim()
    links([(0, 1)], ['p'], bases={0: 99, 1: 139}, orient='horizontal', ax=ax)
    assert ax.get_ylim() == before


def test_vertical_does_not_touch_xlim(ax):
    _draw_boxes(ax)
    before = ax.get_xlim()
    links([(0, 1)], ['p'], bases={0: 99, 1: 139}, ax=ax)
    assert ax.get_xlim() == before


def test_default_base_is_the_axis_boundary_in_the_growth_direction(ax):
    _draw_boxes(ax, 'horizontal')
    xlim_right = ax.get_xlim()[1]
    result = links([(0, 1)], ['p'], orient='horizontal', ax=ax)
    assert result.links[0].feet == (xlim_right, xlim_right)


def test_horizontal_levels_pack_like_vertical(ax):
    result = links([(0, 2), (0, 1)], ['0 vs 2', '0 vs 1'], bases={0: 100, 1: 50, 2: 140}, orient='horizontal', ax=ax)
    assert result.links[0].bar_value > result.links[1].bar_value  # the wider span is pushed up a level


# ----- orient values -----


@pytest.mark.parametrize(
    'orient, expected',
    [('vertical', 'vertical'), ('v', 'vertical'), ('V', 'vertical'), ('horizontal', 'horizontal'), ('h', 'horizontal'), ('H', 'horizontal')],
)
def test_orient_spellings(ax, orient, expected):
    assert links([(0, 1)], ['p'], bases={0: 1, 1: 2}, orient=orient, ax=ax).orient == expected


@pytest.mark.parametrize('orient', ['sideways', '', None, 1])
def test_invalid_orient_raises(ax, orient):
    with pytest.raises(ValueError, match='orient'):
        links([(0, 1)], ['p'], orient=orient, ax=ax)


# ----- anchors input -----


def _bar_values(result):
    return [link.bar_value for link in result.links]


def test_anchors_accept_lists_arrays_and_dataframes_equally():
    pairs = [(0, 2), (0, 1), (1, 2)]
    outputs = []
    for anchors in (pairs, np.array(pairs), pd.DataFrame(pairs, columns=['a', 'b']), [list(p) for p in pairs]):
        _, axes = plt.subplots()
        outputs.append(_bar_values(links(anchors, list('xyz'), bases={0: 10, 1: 20, 2: 30}, ax=axes)))
    assert all(output == outputs[0] for output in outputs)


def test_anchor_pair_order_does_not_matter():
    outputs = []
    for anchors in ([(0, 1)], [(1, 0)]):
        _, axes = plt.subplots()
        outputs.append(_bar_values(links(anchors, ['p'], bases={0: 10, 1: 20}, ax=axes)))
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize(
    'anchors',
    [[0, 1], [(0, 1, 2)], np.zeros((2, 3)), [(0, 1), (2,)], pd.DataFrame({'a': [0], 'b': [1], 'c': [2]})],
    ids=['flat', 'triple', 'n_by_3', 'ragged', 'three_columns'],
)
def test_bad_anchor_shape_raises(ax, anchors):
    with pytest.raises(ValueError, match='anchors'):
        links(anchors, ['p'], ax=ax)


@pytest.mark.parametrize('anchors', [[(0, np.nan)], [(0, np.inf)], [('a', 'b')]], ids=['nan', 'inf', 'strings'])
def test_non_numeric_or_non_finite_anchors_raise(ax, anchors):
    with pytest.raises(ValueError, match='anchors'):
        links(anchors, ['p'], ax=ax)


def test_empty_anchors_draw_nothing(ax):
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    result = links([], [], ax=ax)
    assert result.links == [] and result.levels == []
    assert (ax.get_xlim(), ax.get_ylim()) == (xlim, ylim)


def test_texts_and_colors_must_match_anchors(ax):
    with pytest.raises(ValueError, match='texts'):
        links([(0, 1), (1, 2)], ['only one'], ax=ax)
    with pytest.raises(ValueError, match='colors'):
        links([(0, 1)], ['p'], colors=['red', 'blue'], ax=ax)


def test_options_are_keyword_only(ax):
    with pytest.raises(TypeError):
        links([(0, 1)], ['p'], {0: 1, 1: 2}, ax=ax)


# ----- bases input -----


def test_bases_accepts_a_series(ax):
    result = links([(0, 1)], ['p'], bases=pd.Series({0: 99.0, 1: 139.0}), ax=ax)
    assert result.links[0].feet == (99.0, 139.0)


def test_result_bases_are_native_python_types(ax):
    result = links(np.array([(0, 1)]), ['p'], bases={np.int64(0): np.float64(5), np.int64(1): np.float64(6)}, ax=ax)
    assert all(type(key) is int for key in result.bases)
    assert all(type(value) is float for value in result.bases.values())


# ----- placement information -----


def test_levels_and_link_metadata(ax):
    result = links([(0, 2), (0, 1), (2, 3)], ['wide', 'a', 'b'], bases={0: 10, 1: 10, 2: 10, 3: 10}, ax=ax)

    assert result.levels == [1, 0, 0]  # the narrow links share level 0, the one spanning them is above
    assert [link.level for link in result.links] == result.levels
    assert [link.anchors for link in result.links] == [(0, 2), (0, 1), (2, 3)]
    for link in result.links:
        assert isinstance(link, LinkArtists)
        assert link.bar_top > link.bar_value > max(link.feet)
    assert result.links[0].bar_value > result.links[1].bar_top  # a higher level clears the lower one


def test_artists_can_be_restyled_through_the_result(ax):
    result = links([(0, 1)], ['p'], bases={0: 1, 1: 2}, ax=ax)
    link = result.links[0]
    assert link.text in ax.texts and link.line in ax.lines
    link.text.set_color('red')
    assert link.text.get_color() == 'red'


def test_per_link_colors_apply_to_text_only(ax):
    result = links([(0, 1), (1, 2)], ['a', 'b'], bases={0: 1, 1: 2, 2: 3}, colors=['red', 'blue'], ax=ax)
    assert [link.text.get_color() for link in result.links] == ['red', 'blue']


def test_final_bases_rise_over_spanned_positions(ax):
    result = links([(0, 2)], ['p'], bases={0: 10, 1: 50, 2: 20}, ax=ax)
    assert result.bases[0] == result.bases[1] == result.bases[2] == result.links[0].bar_top
    assert result.bases[1] > 50
