import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import seaborn as sns
from matplotlib import colors as mpl_colors

from aa_utilities.graphics import heatmap, overlay_boxes

plt.switch_backend('Agg')


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.normal(size=(4, 5)), index=list('abcd'), columns=list('vwxyz'))


@pytest.fixture
def clustermap(data):
    return sns.clustermap(data)


def _box_sizes(patch_collection):
    return np.array([np.ptp(path.vertices[:, 0]) for path in patch_collection.get_paths()])


# ----- scalar broadcasting -----


@pytest.mark.parametrize('size', [0.5, 1.5])  # sizes above 1 are allowed, boxes then overlap
def test_scalar_sizes_apply_to_every_cell(clustermap, size):
    boxes = overlay_boxes(clustermap, sizes=size)
    assert len(boxes.get_paths()) == 20
    assert np.allclose(_box_sizes(boxes), size)


@pytest.mark.parametrize('color', ['black', (1, 0, 0), (1, 0, 0, 0.5)], ids=['name', 'rgb', 'rgba'])
def test_scalar_facecolors_apply_to_every_cell(clustermap, color):
    boxes = overlay_boxes(clustermap, facecolors=color)
    assert np.allclose(boxes.get_facecolor(), mpl_colors.to_rgba(color))


def test_scalar_edgecolors_and_linewidths_apply_to_every_cell(clustermap):
    boxes = overlay_boxes(clustermap, edgecolors='red', linewidths=3)
    assert np.allclose(boxes.get_edgecolor(), mpl_colors.to_rgba('red'))
    assert np.allclose(boxes.get_linewidth(), 3)


def test_defaults_are_unchanged(clustermap):
    boxes = overlay_boxes(clustermap)
    assert np.allclose(_box_sizes(boxes), 0.8)
    assert np.allclose(boxes.get_linewidth(), 1.5)
    assert len(boxes.get_edgecolor()) == 0 or np.allclose(boxes.get_edgecolor()[:, 3], 0)


# ----- full matrices keep working -----


def test_matrices_follow_the_original_data_order(data, clustermap):
    sizes = np.linspace(0.1, 0.9, data.size).reshape(data.shape)
    boxes = overlay_boxes(clustermap, sizes=sizes)

    row_order = clustermap.dendrogram_row.reordered_ind
    col_order = clustermap.dendrogram_col.reordered_ind
    expected = sizes[np.ix_(row_order, col_order)].ravel()  # patches are drawn row by row in visual order
    assert np.allclose(_box_sizes(boxes), expected)


def test_color_matrices_with_and_without_channels(data, clustermap):
    names = np.full(data.shape, 'blue')
    assert np.allclose(overlay_boxes(clustermap, facecolors=names).get_facecolor(), mpl_colors.to_rgba('blue'))

    rgba = np.tile(mpl_colors.to_rgba('green'), (*data.shape, 1))
    assert np.allclose(overlay_boxes(sns.clustermap(data), facecolors=rgba).get_facecolor(), rgba[0, 0])


# ----- validation -----


@pytest.mark.parametrize(
    'kwargs, name',
    [
        ({'sizes': np.ones((5, 4))}, 'sizes'),
        ({'sizes': 'large'}, 'sizes'),
        ({'linewidths': np.ones(5)}, 'linewidths'),
        ({'facecolors': 'not-a-color'}, 'facecolors'),
        ({'facecolors': 0.5}, 'facecolors'),
        ({'edgecolors': np.full((5, 4), 'red')}, 'edgecolors'),
    ],
)
def test_invalid_values_raise_a_clear_error(clustermap, kwargs, name):
    with pytest.raises(ValueError, match=name):
        overlay_boxes(clustermap, **kwargs)


# ----- heatmap() -----


def test_heatmap_accepts_scalar_box_options(data):
    fig = heatmap(data, {'sizes': 0.6, 'edgecolors': 'black', 'linewidths': 2})
    heat_ax = next(ax for ax in fig.axes if ax.get_label() == 'heatmap')
    boxes = heat_ax.collections[-1]
    assert np.allclose(_box_sizes(boxes), 0.6)
    assert np.allclose(boxes.get_edgecolor(), mpl_colors.to_rgba('black'))
    assert np.allclose(boxes.get_linewidth(), 2)
