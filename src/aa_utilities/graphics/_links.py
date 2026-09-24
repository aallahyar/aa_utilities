"""Draws comparison "links" (a bracket connecting two x-positions with a text label above
it, e.g. a p-value) on a matplotlib Axes, optionally stacking several of them so they don't
overlap.
"""

from collections import namedtuple
from dataclasses import dataclass, field

from matplotlib import pyplot as plt, transforms as mpl_transforms

from ._utilities import _pixel_offset_to_data, _bbox_to_data

LinkArtists = namedtuple('LinkArtists', ['line', 'text', 'y_top'])


@dataclass
class LinksResult:
    """Result of `links()`. A plain object (not a tuple) so new fields can be added later
    without breaking existing attribute-based usage (`result.links`, `result.y_bases`, ...).
    """

    links: list = field(default_factory=list)
    y_bases: dict = field(default_factory=dict)


def _link(
    x_left,
    x_right,
    text,
    y_left,
    y_right=None,
    y_top=None,
    height=10,
    pad=5,
    units='points',
    ax=None,
    line_kws=None,
    text_kws=None,
):
    """Draws a single comparison bracket between `x_left` and `x_right`, with `text` placed
    above it. Pure drawing primitive: never mutates `ax`'s limits.

    Parameters:
    ----------
    x_left, x_right: float
        x-positions (e.g. x-tick indices) to link.
    text: str
        text placed over the link's line.
    y_left, y_right: float
        y-positions where the bracket's vertical arms start (`y_right` defaults to `y_left`).
    y_top: float, optional
        y-position of the bracket's horizontal bar. If not given, it is placed `height`
        `units` above `max(y_left, y_right)`.
    height: float
        vertical extent of the bracket's arms, in `units`, when `y_top` is not given.
    pad: float
        gap between `y_left`/`y_right`/`y_top` and where the bracket is actually drawn, in
        `units`.
    units: str
        units for `height`/`pad` (and for `links()`'s `top_space`). `'points'` (default) is
        resolution-independent (a fixed physical size); `'dots'` is a fixed pixel count,
        which looks different physical sizes at different figure DPIs.
    ax: matplotlib Axes, optional
    line_kws, text_kws: dict, optional
        passed to `ax.plot()`/`ax.text()` respectively.

    Returns:
    -------
    LinkArtists(line, text, y_top)
    """
    if ax is None:
        ax = plt.gca()
    if y_right is None:
        y_right = y_left
    if line_kws is None:
        line_kws = {'linestyle': '-', 'color': '#555555', 'linewidth': 0.5}
    if text_kws is None:
        text_kws = {'color': '#555555'}

    if y_top is None:
        y_top = _pixel_offset_to_data(ax, max(y_left, y_right), offset=height, units=units)

    transform_pad = mpl_transforms.offset_copy(ax.transData, y=pad, units=units, fig=ax.get_figure())

    line_artist = ax.plot(
        [x_left, x_left, x_right, x_right],
        [y_left, y_top, y_top, y_right],
        transform=transform_pad,
        **line_kws,
    )[0]
    text_artist = ax.text(
        (x_left + x_right) / 2,
        y_top,
        text,
        transform=transform_pad,
        va='bottom',
        ha='center',
        **text_kws,
    )

    return LinkArtists(line=line_artist, text=text_artist, y_top=y_top)


def links(
    x_left,
    x_right,
    text,
    y_bases=None,
    auto_order=True,
    height=10,
    pad=5,
    top_space=10,
    units='points',
    ax=None,
    line_kws=None,
    text_kws=None,
):
    """Draws one or more comparison links, automatically stacking overlapping ones onto
    different vertical levels so a link's bar always clears any position/link it visually
    spans over, and expands the y-axis at most once to fit them all.

    Levels are assigned via a greedy interval-packing algorithm: two links share a level only
    if their `[x_left, x_right]` spans don't overlap; a link's bar height then only needs to
    clear the real data in its own span plus any lower-level link whose span overlaps it - an
    x-position's true height (from `y_bases`) is never overwritten by an unrelated link merely
    passing over it, so the result doesn't depend on the order links are given in.

    Parameters:
    ----------
    x_left, x_right: array-like
        x-positions to link, one entry per link.
    text: array-like of str
        text placed above each link's bar, one entry per link.
    y_bases: Mapping, optional
        x-position -> its real minimum height (e.g. a group's own data max). Any position not
        listed here defaults to the current `ax.get_ylim()[1]`.
    auto_order: bool
        if True (default), links are packed into levels narrowest-span-first, which tends to
        minimize the number of levels used. If False, levels are assigned in the given input
        order instead - this never reintroduces the flattening issue above, but may use more
        levels (i.e. more stacking) than necessary.
    height: float
        vertical extent of a link's arms, in `units`, from its feet up to its horizontal bar.
    pad: float
        gap between a link's nominal y-values and where it is actually drawn, in `units` -
        keeps the bracket from visually touching the data/position it starts from.
    top_space: float
        gap (in `units`) reserved above a link's rendered text before a higher-level,
        overlapping link (or the y-axis boundary) may start.
    units: str
        units for `height`/`pad`/`top_space`. `'points'` (default) is resolution-independent
        (a fixed physical size); `'dots'` is a fixed pixel count, which looks like a different
        physical size at different figure DPIs.
    ax: matplotlib Axes, optional
    line_kws, text_kws: dict, optional
        passed to `ax.plot()`/`ax.text()` respectively, for styling the bracket/text.

    Returns:
    -------
    LinksResult(links, y_bases)
        `links`: list[LinkArtists], one per link, in input order.
        `y_bases`: per-position ceiling reached by anything touching that position (its own
        `y_bases` value, or the top of the tallest link spanning over it, whichever is higher).
    """
    if ax is None:
        ax = plt.gca()

    x_left = list(x_left)
    x_right = list(x_right)
    text = list(text)
    n = len(x_left)
    if not (len(x_right) == len(text) == n):
        raise ValueError(
            f'x_left, x_right, and text must all have the same length. Got {len(x_left)}, {len(x_right)}, {len(text)}.'
        )

    given_bases = dict(y_bases) if y_bases is not None else {}
    default_base = ax.get_ylim()[1]
    positions = sorted(set(x_left) | set(x_right) | set(given_bases))
    position_index = {p: i for i, p in enumerate(positions)}
    natural_base = {p: given_bases.get(p, default_base) for p in positions}

    spans = [tuple(sorted((position_index[x_left[i]], position_index[x_right[i]]))) for i in range(n)]

    def overlaps(a, b):
        # inclusive: links that merely touch at a shared endpoint still can't share a level,
        # since one's foot would otherwise sit right on top of the other's text
        return max(a[0], b[0]) <= min(a[1], b[1])

    # assign each link to the lowest level whose already-placed spans don't overlap it
    assignment_order = sorted(range(n), key=lambda i: (spans[i][1] - spans[i][0], i)) if auto_order else range(n)
    level_of = {}
    level_spans = []
    for i in assignment_order:
        level = next((lv for lv, spans_at_level in enumerate(level_spans) if not any(overlaps(spans[i], s) for s in spans_at_level)), None)
        if level is None:
            level = len(level_spans)
            level_spans.append([])
        level_spans[level].append(spans[i])
        level_of[i] = level

    # compute each link's feet/bar height in level order, so lower levels are already known.
    # a position's "ceiling" is its real base, raised to clear any *lower-level* bracket
    # whose span covers it - this keeps a link's own arms from visually crossing through a
    # lower bracket it shares an endpoint with, while leaving same-level neighbors alone.
    results = [None] * n
    covered = []  # (level, span, bar_top) for links already processed
    final_heights = dict(natural_base)
    overall_top = None
    top_text_artist = None

    def ceiling(level, position_idx):
        value = natural_base[positions[position_idx]]
        for lower_level, (lo_j, hi_j), bar_top_j in covered:
            if lower_level < level and lo_j <= position_idx <= hi_j:
                value = max(value, bar_top_j)
        return value

    for i in sorted(range(n), key=lambda i: (level_of[i], i)):
        lo, hi = spans[i]
        level = level_of[i]
        clearance_ref = max(ceiling(level, p) for p in range(lo, hi + 1))
        foot_left = ceiling(level, position_index[x_left[i]])
        foot_right = ceiling(level, position_index[x_right[i]])

        artists = _link(
            x_left=x_left[i],
            x_right=x_right[i],
            text=text[i],
            y_left=foot_left,
            y_right=foot_right,
            y_top=_pixel_offset_to_data(ax, clearance_ref, offset=height, units=units),
            pad=pad,
            units=units,
            ax=ax,
            line_kws=line_kws,
            text_kws=text_kws,
        )
        results[i] = artists

        text_extent = _bbox_to_data(ax, artists.text.get_window_extent())
        bar_top = _pixel_offset_to_data(ax, text_extent.top, offset=top_space, units=units)
        covered.append((level, spans[i], bar_top))
        for p in positions[lo : hi + 1]:
            final_heights[p] = max(final_heights[p], bar_top)
        if overall_top is None or bar_top > overall_top:
            overall_top = bar_top
            top_text_artist = artists.text


    if overall_top is not None:
        ax.set_ylim(top=max(overall_top, ax.get_ylim()[1]))

        # changing ylim (especially on a log scale) shifts the data<->pixel mapping used
        # above, so the topmost text's clearance must be re-checked (and re-grown if needed)
        # against the *new* scale - otherwise it can end up poking past the new axis top.
        for _ in range(4):
            text_extent = _bbox_to_data(ax, top_text_artist.get_window_extent())
            required_top = _pixel_offset_to_data(ax, text_extent.top, offset=top_space, units=units)
            if required_top <= ax.get_ylim()[1]:
                break
            ax.set_ylim(top=required_top)

    return LinksResult(links=results, y_bases=final_heights)


# %%
if __name__ == '__main__':
    import numpy as np

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # no y_bases given: every position starts at the current axis top
    ax1.boxplot(x=[np.linspace(1, 100), np.linspace(40, 140)], positions=[0, 1])
    ax1.set_yscale('log', base=10)
    n_links = 6
    links(
        x_left=[0] * n_links,
        x_right=[1] * n_links,
        text=[f'p-value {i}' for i in range(n_links)],
        ax=ax1,
    )

    # explicit y_bases: an untouched, tall middle position is still respected
    ax2.boxplot(
        x=[np.linspace(1, 100), np.linspace(10, 500), np.linspace(40, 140)],
        positions=[0, 1, 2],
    )
    ax2.set_yscale('log', base=10)
    result = links(
        x_left=[0, 0],
        x_right=[2, 1],
        text=['0 vs 2', '0 vs 1'],
        y_bases={0: 100, 1: 500, 2: 140},
        ax=ax2,
    )
    print('final y_bases:', result.y_bases)

    # order-independence: (1-2) then (1-3), or (1-3) then (1-2), give the same layout
    fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(6, 4))
    for ax, order in [(ax3, ['A', 'B']), (ax4, ['B', 'A'])]:
        ax.boxplot(x=[np.linspace(1, 10)] * 3, positions=[1, 2, 3])
        pairs = {'A': (1, 2), 'B': (1, 3)}
        result = links(
            x_left=[pairs[name][0] for name in order],
            x_right=[pairs[name][1] for name in order],
            text=order,
            y_bases={1: 10, 2: 5, 3: 10},
            ax=ax,
        )
        ax.set_title(f'input order: {order}')

    plt.show()


