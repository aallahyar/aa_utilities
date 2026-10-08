import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import seaborn as sns
from matplotlib import colors as mpl_colors

from aa_utilities.graphics import MultilevelTicks, draw_multilevel_ticks

plt.switch_backend('Agg')


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')


@pytest.fixture
def columns():
    return pd.MultiIndex.from_product([['2023', '2024'], ['Q1', 'Q2', 'Q3'], ['alpha', 'beta']], names=['y', 'q', 'm'])


@pytest.fixture
def rows():
    return pd.MultiIndex.from_product([['Placebo', 'Treated'], ['Low', 'High'], ['U', 'V']])


@pytest.fixture
def frame(rows, columns):
    return pd.DataFrame(np.random.default_rng(0).normal(size=(len(rows), len(columns))), index=rows, columns=columns)


def _heatmap(frame, figsize=(9, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(frame, ax=ax, cbar=False)
    return fig, ax


def _group_texts(ax):
    return [text for text in ax.texts if text.get_text()]


def _separators(ax):
    return [text for text in ax.texts if not text.get_text()]


# ----- content and positions -----


def test_columns_get_inner_tick_labels_and_one_row_of_text_per_outer_level(frame):
    _, ax = _heatmap(frame)
    draw_multilevel_ticks(ax, columns=frame.columns)

    assert [label.get_text() for label in ax.get_xticklabels()] == ['alpha', 'beta'] * 6
    assert [text.get_text() for text in _group_texts(ax)] == ['Q1', 'Q2', 'Q3'] * 2 + ['2023', '2024']  # level 1 then 0


def test_group_labels_are_centred_on_their_span(frame):
    _, ax = _heatmap(frame)
    draw_multilevel_ticks(ax, columns=frame.columns)

    centres = {text.get_text(): text.xy[0] for text in _group_texts(ax) if text.get_text() in ('2023', '2024')}
    assert centres == {'2023': 3.0, '2024': 9.0}  # spans of 6 cells, with cell centres at i + 0.5


def test_cell_offset_shifts_labels_for_imshow_style_axes(frame):
    _, ax = _heatmap(frame)
    draw_multilevel_ticks(ax, columns=frame.columns, cell_offset=0)

    centres = {text.get_text(): text.xy[0] for text in _group_texts(ax) if text.get_text() in ('2023', '2024')}
    assert centres == {'2023': 2.5, '2024': 8.5}


def test_one_separator_between_each_pair_of_neighbouring_groups(frame):
    _, ax = _heatmap(frame)
    draw_multilevel_ticks(ax, columns=frame.columns)

    assert len(_separators(ax)) == 5 + 1  # 6 quarter groups, 2 year groups
    assert sorted({round(text.xy[0], 6) for text in _separators(ax)}) == [2, 4, 6, 8, 10]


def test_rows_use_the_left_axis_with_rotated_group_labels(frame):
    _, ax = _heatmap(frame)
    draw_multilevel_ticks(ax, index=frame.index)

    assert [label.get_text() for label in ax.get_yticklabels()] == ['U', 'V'] * 4
    outer = [text for text in _group_texts(ax) if text.get_text() in ('Placebo', 'Treated')]
    assert {text.get_text(): text.xy[1] for text in outer} == {'Placebo': 2.0, 'Treated': 6.0}
    assert all(text.get_rotation() == 90 for text in outer)


def test_rows_and_columns_together(frame):
    _, ax = _heatmap(frame)
    draw_multilevel_ticks(ax, index=frame.index, columns=frame.columns)
    assert len(_group_texts(ax)) == (6 + 2) + (4 + 2)


def test_rotation_applies_to_the_column_labels_only(frame):
    _, ax = _heatmap(frame)
    draw_multilevel_ticks(ax, index=frame.index, columns=frame.columns, rotation=45)
    assert ax.get_xticklabels()[0].get_rotation() == 45
    assert ax.get_yticklabels()[0].get_rotation() == 0


def test_more_levels_than_three():
    index = pd.MultiIndex.from_product([['A', 'B'], ['a', 'b'], ['i', 'ii'], ['x', 'y']])
    _, ax = _heatmap(pd.DataFrame(np.ones((len(index), 3)), index=index))
    draw_multilevel_ticks(ax, index=index)
    assert len(_group_texts(ax)) == 2 + 4 + 8


def test_non_contiguous_groups_show_up_as_separate_segments():
    index = pd.MultiIndex.from_arrays([['a', 'b', 'a'], ['x', 'y', 'z']])
    _, ax = _heatmap(pd.DataFrame(np.ones((3, 2)), index=index))
    draw_multilevel_ticks(ax, index=index)
    assert [text.get_text() for text in _group_texts(ax)] == ['a', 'b', 'a']


# ----- layout -----


def _level_boxes(ax):
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outer = {name: [t.get_window_extent(renderer) for t in _group_texts(ax) if t.get_text() in names]
             for name, names in {'quarter': ('Q1', 'Q2', 'Q3'), 'year': ('2023', '2024')}.items()}
    ticks = [label.get_window_extent(renderer) for label in ax.get_xticklabels()]
    return ticks, outer['quarter'], outer['year']


@pytest.mark.parametrize('figsize, rotation', [((9, 6), 0), ((9, 2.5), 0), ((9, 5), 90), ((9, 5), 45)])
def test_levels_never_overlap_the_inner_tick_labels_or_each_other(frame, figsize, rotation):
    _, ax = _heatmap(frame, figsize)
    draw_multilevel_ticks(ax, columns=frame.columns, rotation=rotation)

    ticks, quarters, years = _level_boxes(ax)
    assert max(box.y1 for box in quarters) <= min(box.y0 for box in ticks) + 0.5
    assert max(box.y1 for box in years) <= min(box.y0 for box in quarters) + 0.5


def test_spacing_is_independent_of_dpi(frame):
    gaps = []
    for dpi in (100, 200):
        fig, ax = _heatmap(frame)
        fig.set_dpi(dpi)
        draw_multilevel_ticks(ax, columns=frame.columns)
        ticks, quarters, _ = _level_boxes(ax)
        gaps.append((min(box.y0 for box in ticks) - max(box.y1 for box in quarters)) * 72 / dpi)
    assert gaps[0] == pytest.approx(gaps[1], abs=0.5)


def test_label_and_line_options_are_applied(frame):
    _, ax = _heatmap(frame)
    draw_multilevel_ticks(ax, columns=frame.columns, label_kws={'fontweight': 'bold'}, line_kws={'color': 'red'})
    assert all(text.get_fontweight() == 'bold' for text in _group_texts(ax))
    assert all(text.arrow_patch.get_edgecolor()[:3] == (1.0, 0.0, 0.0) for text in _separators(ax))


# ----- behaviour -----


def test_plain_index_only_gets_ordinary_tick_labels():
    _, ax = _heatmap(pd.DataFrame(np.ones((3, 2)), index=list('abc'), columns=['x', 'y']))
    draw_multilevel_ticks(ax, index=pd.Index(list('abc')), columns=pd.Index(['x', 'y']))
    assert [label.get_text() for label in ax.get_yticklabels()] == ['a', 'b', 'c']
    assert not ax.texts


def test_the_flattened_axis_label_is_cleared(frame):
    _, ax = _heatmap(frame)
    assert ax.get_xlabel()  # seaborn names it after the levels
    draw_multilevel_ticks(ax, columns=frame.columns)
    assert ax.get_xlabel() == ''


def test_calling_twice_does_not_stack_labels(frame):
    _, ax = _heatmap(frame)
    draw_multilevel_ticks(ax, index=frame.index, columns=frame.columns)
    n_artists = len(ax.texts)
    draw_multilevel_ticks(ax, index=frame.index, columns=frame.columns)
    assert len(ax.texts) == n_artists


def test_axis_limits_are_untouched(frame):
    _, ax = _heatmap(frame)
    limits = (ax.get_xlim(), ax.get_ylim())
    draw_multilevel_ticks(ax, index=frame.index, columns=frame.columns)
    assert (ax.get_xlim(), ax.get_ylim()) == limits


def test_nothing_to_draw_raises(frame):
    _, ax = _heatmap(frame)
    with pytest.raises(ValueError, match='index'):
        draw_multilevel_ticks(ax)


# ----- returned artists (`ax.multilevel_ticks`) -----


def _texts(artists):
    return [artist.get_text() for artist in artists]


def test_returns_the_artists_and_stores_them_on_the_axes(frame):
    _, ax = _heatmap(frame)
    ticks = draw_multilevel_ticks(ax, index=frame.index, columns=frame.columns)

    assert isinstance(ticks, MultilevelTicks)
    assert ax.multilevel_ticks is ticks


def test_artists_are_grouped_per_level_following_pandas_level_numbers(frame):
    _, ax = _heatmap(frame)
    ticks = draw_multilevel_ticks(ax, index=frame.index, columns=frame.columns)

    assert [_texts(level) for level in ticks.columns.labels] == [['2023', '2024'], ['Q1', 'Q2', 'Q3'] * 2]
    assert [_texts(level) for level in ticks.index.labels] == [['Placebo', 'Treated'], ['Low', 'High'] * 2]
    assert [len(level) for level in ticks.columns.separators] == [1, 5]
    assert [len(level) for level in ticks.index.separators] == [1, 3]


def test_plain_index_has_no_outer_levels():
    _, ax = _heatmap(pd.DataFrame(np.ones((3, 2)), index=list('abc'), columns=['x', 'y']))
    ticks = draw_multilevel_ticks(ax, index=pd.Index(list('abc')))
    assert ticks.index.labels == [] and ticks.index.separators == []


def test_default_lines_are_light_gray_and_thin(frame):
    _, ax = _heatmap(frame)
    ticks = draw_multilevel_ticks(ax, columns=frame.columns)
    patch = ticks.columns.separators[0][0].arrow_patch
    assert patch.get_edgecolor()[:3] == pytest.approx(mpl_colors.to_rgb('#aaaaaa'))
    assert patch.get_linewidth() == pytest.approx(0.6)


def test_artists_can_be_restyled_through_the_container(frame):
    _, ax = _heatmap(frame)
    ticks = draw_multilevel_ticks(ax, columns=frame.columns)
    for separator in ticks.columns.separators[0]:
        separator.arrow_patch.set_linestyle('--')
    assert ticks.columns.separators[0][0].arrow_patch.get_linestyle() == '--'


def test_remove_takes_everything_off_the_axes_and_can_be_repeated(frame):
    _, ax = _heatmap(frame)
    ticks = draw_multilevel_ticks(ax, index=frame.index, columns=frame.columns)
    ticks.remove()
    assert not ax.texts
    ticks.remove()  # artists that are already gone are skipped


def test_redrawing_after_a_manual_removal_works(frame):
    _, ax = _heatmap(frame)
    ticks = draw_multilevel_ticks(ax, columns=frame.columns)
    ticks.columns.labels[0][0].remove()
    draw_multilevel_ticks(ax, columns=frame.columns)
    assert len(_group_texts(ax)) == 8


def test_redrawing_one_axis_keeps_the_other(frame):
    _, ax = _heatmap(frame)
    first = draw_multilevel_ticks(ax, index=frame.index)
    rows = first.index
    second = draw_multilevel_ticks(ax, columns=frame.columns)

    assert second is first and second.index is rows
    assert all(artist in ax.texts for level in rows.labels for artist in level)
    assert len(_group_texts(ax)) == (4 + 2) + (6 + 2)


# ----- panels with different sizes -----


@pytest.mark.parametrize('layout', ['constrained', 'tight'])
def test_panels_with_different_size_ratios_get_the_same_spacing(layout):
    fig = plt.figure(figsize=(12, 7), layout=layout)
    grid = fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[1, 2])
    rng = np.random.default_rng(0)
    panels, ticks_per_panel = [], []
    for cell, (n_years, n_rows) in zip(grid, [(2, 1), (1, 1), (2, 2), (1, 2)]):
        cols = pd.MultiIndex.from_product([[str(y) for y in range(n_years)], ['Q1', 'Q2', 'Q3'], ['a', 'b']])
        rows = pd.MultiIndex.from_product([['P', 'T'], [f'g{i}' for i in range(n_rows)], ['U', 'V']])
        ax = fig.add_subplot(cell)
        sns.heatmap(pd.DataFrame(rng.normal(size=(len(rows), len(cols))), index=rows, columns=cols), ax=ax, cbar=False)
        ticks_per_panel.append(draw_multilevel_ticks(ax, index=rows, columns=cols))
        panels.append(ax)
    if layout == 'tight':
        fig.tight_layout()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    for ax, ticks in zip(panels, ticks_per_panel):
        inner = min(label.get_window_extent(renderer).y0 for label in ax.get_xticklabels())
        quarters = max(text.get_window_extent(renderer).y1 for text in ticks.columns.labels[1])
        assert (inner - quarters) * 72 / fig.dpi == pytest.approx(4.0, abs=0.5)  # the default `pad`
        every_artist = [a for axis in (ticks.columns, ticks.index) for level in axis.labels for a in level]
        assert all(fig.bbox.padded(1).contains(*a.get_window_extent(renderer).p0) for a in every_artist)


# ----- other sides (clustermap rows on the right, column ticks on top) -----


def test_clustermap_rows_get_their_levels_on_the_right(frame):
    grid = sns.clustermap(frame, row_cluster=False, col_cluster=False)
    ax = grid.ax_heatmap
    ticks = draw_multilevel_ticks(ax, index=grid.data2d.index, columns=grid.data2d.columns)

    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    heat_right = ax.get_window_extent(renderer).x1
    outer = [text for level in ticks.index.labels for text in level]
    assert all(text.xy[0] == 1 for text in outer)  # anchored at the axes' right edge
    assert all(text.get_window_extent(renderer).x0 >= heat_right for text in outer)
    assert all(label.get_window_extent(renderer).x1 <= text.get_window_extent(renderer).x0 + 0.5
               for label in ax.get_yticklabels() for text in ticks.index.labels[1])


def test_column_labels_on_top_stack_upwards(frame):
    _, ax = _heatmap(frame)
    ax.xaxis.tick_top()
    ticks = draw_multilevel_ticks(ax, columns=frame.columns)

    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    axes_top = ax.get_window_extent(renderer).y1
    inner = max(label.get_window_extent(renderer).y1 for label in ax.get_xticklabels())
    quarters = [t.get_window_extent(renderer) for t in ticks.columns.labels[1]]
    years = [t.get_window_extent(renderer) for t in ticks.columns.labels[0]]
    assert min(box.y0 for box in quarters) >= inner - 0.5 > axes_top
    assert min(box.y0 for box in years) >= max(box.y1 for box in quarters) - 0.5

