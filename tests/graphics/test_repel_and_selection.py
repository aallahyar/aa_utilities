import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pytest

from aa_utilities.graphics import InteractiveSelection, repel_text


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')


@pytest.mark.parametrize('bad', [np.nan, np.inf])
def test_repel_text_rejects_non_finite_anchors(bad):
    _, ax = plt.subplots()
    ax.scatter([0, 1], [0, 1])
    with pytest.raises(ValueError, match='finite'):
        repel_text(ax, [0, bad], [0, 1], ['a', 'b'])


def test_repel_text_places_one_label_per_anchor():
    _, ax = plt.subplots()
    ax.scatter([0, 0.01], [0, 0.01])
    labels = repel_text(ax, [0, 0.01], [0, 0.01], ['a', 'b'])
    assert len(labels) == 2


@pytest.fixture
def interactive_backend(monkeypatch):
    monkeypatch.setattr(matplotlib, 'get_backend', lambda: 'ipympl')  # the tool refuses other backends


def test_interactive_selection_uses_the_axes_of_its_collection(interactive_backend):
    _, ax_collection = plt.subplots()
    collection = ax_collection.scatter([0, 1, 2], [0, 1, 2])
    _, other_ax = plt.subplots()  # now the current axes

    selection = InteractiveSelection(collection)

    assert selection.ax is ax_collection


def test_interactive_selection_rejects_axes_that_do_not_hold_the_collection(interactive_backend):
    _, ax_collection = plt.subplots()
    collection = ax_collection.scatter([0, 1, 2], [0, 1, 2])
    _, other_ax = plt.subplots()

    with pytest.raises(ValueError, match='axes'):
        InteractiveSelection(collection, ax=other_ax)
