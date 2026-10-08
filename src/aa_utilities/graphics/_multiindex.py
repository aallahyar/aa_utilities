from dataclasses import dataclass, field
from itertools import chain

import numpy as np
import pandas as pd
from matplotlib.text import Annotation
from matplotlib.transforms import Bbox

_POINTS_PER_INCH = 72


@dataclass
class AxisLevels:
    """Artists drawn for the outer levels of one axis.

    `labels[level]` and `separators[level]` follow pandas' level numbers (0 is the outermost level).
    The innermost level is the axis' ordinary tick labels, so it is not listed here.
    """

    labels: list[list[Annotation]] = field(default_factory=list)
    separators: list[list[Annotation]] = field(default_factory=list)

    def remove(self):
        """Removes every artist of this axis from its axes (artists that are already gone are skipped)."""
        for artist in chain.from_iterable(chain(self.labels, self.separators)):
            if artist.axes is not None:
                artist.remove()


@dataclass
class MultilevelTicks:
    """Result of `draw_multilevel_ticks()`, also available afterwards as `ax.multilevel_ticks`."""

    columns: AxisLevels | None = None
    index: AxisLevels | None = None

    def remove(self):
        """Removes everything `draw_multilevel_ticks()` drew on the axes."""
        for levels in (self.columns, self.index):
            if levels is not None:
                levels.remove()


def _group_spans(index, level):
    """Returns the start, end (exclusive) and label of each run of identical labels over levels `0..level`."""
    codes = np.array([index.codes[i] for i in range(level + 1)])
    changed = np.any(np.diff(codes, axis=1) != 0, axis=0)
    starts = np.r_[0, np.flatnonzero(changed) + 1]
    ends = np.r_[starts[1:], len(index)]
    return starts, ends, index.get_level_values(level)[starts]


def _to_points(ax, pixels):
    return pixels * _POINTS_PER_INCH / ax.figure.dpi


def _draw_axis(ax, index, axis, cell_offset, rotation, pad, label_kws, line_kws):
    is_x = axis == 'x'

    index = index if isinstance(index, pd.Index) else pd.Index(index)
    labels = [str(label) for label in index.get_level_values(-1)]
    (ax.set_xticks if is_x else ax.set_yticks)(
        np.arange(len(index)) + cell_offset, labels, **({'rotation': rotation} if is_x else {})
    )
    ax.tick_params(axis=axis, length=0)
    (ax.set_xlabel if is_x else ax.set_ylabel)('')  # seaborn names it after the flattened levels
    n_outer = index.nlevels - 1
    drawn = AxisLevels(labels=[[] for _ in range(n_outer)], separators=[[] for _ in range(n_outer)])
    if n_outer == 0 or len(index) == 0:
        return drawn

    # which side the inner tick labels are on, and how far (in points) they reach from the axis
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_box = ax.get_window_extent(renderer)
    tick_labels = ax.get_xticklabels() if is_x else ax.get_yticklabels()
    tick_box = Bbox.union([label.get_window_extent(renderer) for label in tick_labels])
    if is_x:
        outward = 1 if tick_box.y0 + tick_box.y1 > axes_box.y0 + axes_box.y1 else -1  # +1: top, -1: bottom
        reach = (tick_box.y1 - axes_box.y1) if outward > 0 else (axes_box.y0 - tick_box.y0)
    else:
        outward = 1 if tick_box.x0 + tick_box.x1 > axes_box.x0 + axes_box.x1 else -1  # +1: right, -1: left
        reach = (tick_box.x1 - axes_box.x1) if outward > 0 else (axes_box.x0 - tick_box.x0)
    cursor = _to_points(ax, reach) + pad

    transform = ax.get_xaxis_transform() if is_x else ax.get_yaxis_transform()
    if is_x:
        label_defaults = dict(ha='center', va='bottom' if outward > 0 else 'top')
    else:
        label_defaults = dict(ha='left' if outward > 0 else 'right', va='center', rotation=270 if outward > 0 else 90)
    label_kws = label_defaults | (label_kws or {})
    line_kws = dict(color='#aaaaaa', lw=0.6) | (line_kws or {})  # light, so the labels stay in front

    def place(position, distance):
        # (position along the axis, the axes edge it starts from) and the offset away from it, in points
        edge, shift = (1 if outward > 0 else 0), outward * distance
        return ((position, edge), (0, shift)) if is_x else ((edge, position), (shift, 0))

    separators = {}  # level -> (positions of the boundaries, length in points)
    for level in range(n_outer - 1, -1, -1):  # from the level next to the tick labels outwards
        starts, ends, names = _group_spans(index, level)
        for start, end, name in zip(starts, ends, names):
            xy, xytext = place((start + end - 1) / 2 + cell_offset, cursor)
            drawn.labels[level].append(
                ax.annotate(
                    str(name),
                    xy=xy,
                    xycoords=transform,
                    xytext=xytext,
                    textcoords='offset points',
                    annotation_clip=False,
                    **label_kws,
                )
            )
        extents = [text.get_window_extent(renderer) for text in drawn.labels[level]]
        size = _to_points(ax, max(extent.height if is_x else extent.width for extent in extents))
        separators[level] = (ends[:-1] + cell_offset - 0.5, cursor + size + pad / 2)
        cursor += size + pad

    for level, (boundaries, length) in separators.items():
        for boundary in boundaries:
            xy, xytext = place(boundary, length)
            drawn.separators[level].append(
                ax.annotate(
                    '',
                    xy=xy,
                    xycoords=transform,
                    xytext=xytext,
                    textcoords='offset points',
                    annotation_clip=False,
                    arrowprops=dict(arrowstyle='-', shrinkA=0, shrinkB=0, **line_kws),
                )
            )
    return drawn


def draw_multilevel_ticks(ax, index=None, columns=None, *, cell_offset=0.5, rotation=0, pad=4, label_kws=None, line_kws=None):
    """Labels the rows and/or columns of a heatmap with one row of labels per MultiIndex level.

    `sns.heatmap` and similar functions flatten a MultiIndex into a single label (e.g., `2023-Q1-a`).
    This draws the innermost level as the usual tick labels and every outer level beyond it, on the side
    where the tick labels are (below or above the columns, left or right of the rows), centred on its
    group and separated by light lines that extend through the levels
    beneath. Distances are in points, measured from the inner tick labels, so the levels stack without
    overlapping whatever the figure size, panel size or label rotation. Call it again if the inner tick
    labels change afterwards: a repeated call replaces what was drawn before.

    Args:
        ax (matplotlib.axes.Axes): The axes holding the heatmap.
        index (pd.Index, optional): The row labels. Pass the index in the drawn
            order (e.g., `g.data2d.index` for a clustermap). Groups are consecutive runs of equal
            labels: a group that is not contiguous shows up as separate segments.
        columns (pd.Index, optional): The column labels. Same rules as `index`.
        cell_offset (float): Position of a cell's label relative to its integer index: 0.5 for
            `sns.heatmap` and `pcolormesh` (cell centres), 0 for `imshow` and `matshow`.
        rotation (float): Rotation of the innermost column labels. Row labels keep their current rotation.
        pad (float): Space between levels, in points.
        label_kws (dict, optional): Overrides for the outer-level text (passed to `ax.annotate`),
            e.g., `{'fontweight': 'bold'}`.
        line_kws (dict, optional): Overrides for the separator lines, which are light gray and thin by
            default (`color='#aaaaaa'`, `lw=0.6`), e.g., `{'ls': '--'}`.

    Returns:
        MultilevelTicks: The drawn artists, also stored as `ax.multilevel_ticks`. For example,
            `ticks.columns.labels[0]` holds the outermost column labels, `ticks.columns.separators[0]` the
            lines that bound them, and `ticks.remove()` removes everything.

    Notes:
        A plain (single-level) `pd.Index` just gets ordinary tick labels. The axis label is cleared. An
        axis that is not passed keeps the labels seaborn gave it, and the artists drawn for it earlier.

    Example:
        cols = pd.MultiIndex.from_product([['2023', '2024'], ['Q1', 'Q2'], ['a', 'b']])
        df = pd.DataFrame(np.random.default_rng(0).normal(size=(5, len(cols))), columns=cols)
        fig, ax = plt.subplots()
        sns.heatmap(df, ax=ax)
        ticks = draw_multilevel_ticks(ax, columns=df.columns)
    """
    if index is None and columns is None:
        raise ValueError('Provide `index` and/or `columns`.')

    ticks = getattr(ax, 'multilevel_ticks', None)
    if ticks is None:
        ticks = MultilevelTicks()
    for name, labels in (('columns', columns), ('index', index)):
        if labels is None:
            continue
        previous = getattr(ticks, name)
        if previous is not None:
            previous.remove()
        axis = 'x' if name == 'columns' else 'y'
        setattr(ticks, name, _draw_axis(ax, labels, axis, cell_offset, rotation, pad, label_kws, line_kws))

    ax.multilevel_ticks = ticks
    return ticks
