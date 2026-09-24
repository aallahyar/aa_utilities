"""Draws comparison "links" (a bracket connecting two x-positions with a text label above
it, e.g. a p-value) on a matplotlib Axes, optionally stacking several of them so they don't
overlap.
"""

from collections import namedtuple

from matplotlib import pyplot as plt, transforms as mpl_transforms

from ._utilities import _pixel_offset_to_data, _bbox_to_data

LinkArtists = namedtuple('LinkArtists', ['line', 'text', 'y_top'])


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
    y_left,
    y_right=None,
    height=10,
    pad=5,
    top_space=10,
    units='points',
    ax=None,
    line_kws=None,
    text_kws=None,
):
    """Draws one or more comparison links, stacking them (in the given order) so that each
    one clears the previously drawn ones, and expands the y-axis at most once to fit them all.

    `x_left`, `x_right`, `text`, `y_left` (and `y_right`, if given) are array-likes with one
    entry per link (e.g. `pd.Series`/columns of a DataFrame, lists, or numpy arrays) - all
    must have the same length. Pass single-element array-likes to draw just one link.

    Stacking is a simple running "high-water mark": each link's `y_left`/`y_right` is raised
    to at least `top_space` `units` above the previously drawn link's rendered text, if that's
    higher than the link's own given `y_left`/`y_right`. This means links are always stacked
    in input order regardless of `x_left`/`x_right`, i.e. this does not detect whether two
    links' x-ranges actually overlap - pass only genuinely-overlapping/related links together,
    calling `links()` separately per independent group if some don't need to be stacked.

    The y-axis is only ever expanded (never shrunk) to fit the topmost link, so any headroom
    the caller already left in place is preserved.

    Parameters:
    ----------
    height, pad, top_space, units, ax, line_kws, text_kws:
        see `_link()`. `top_space` additionally sets the gap reserved between stacked links.

    Returns:
    -------
    list[LinkArtists]
        one entry per link, in input order.
    """
    if ax is None:
        ax = plt.gca()

    x_left = list(x_left)
    x_right = list(x_right)
    text = list(text)
    y_left = list(y_left)
    n = len(x_left)
    if not (len(x_right) == len(text) == len(y_left) == n):
        raise ValueError(
            'x_left, x_right, text, and y_left must all have the same length. '
            f'Got {len(x_left)}, {len(x_right)}, {len(text)}, {len(y_left)}.'
        )
    if y_right is None:
        y_right = list(y_left)
    else:
        y_right = list(y_right)
        if len(y_right) != n:
            raise ValueError(f'y_right must have the same length as the other arguments. Got {len(y_right)} vs {n}.')

    results = []
    min_next_y = None
    for i in range(n):
        row_y_left = y_left[i]
        row_y_right = y_right[i]
        if min_next_y is not None:
            row_y_left = max(row_y_left, min_next_y)
            row_y_right = max(row_y_right, min_next_y)

        artists = _link(
            x_left=x_left[i],
            x_right=x_right[i],
            text=text[i],
            y_left=row_y_left,
            y_right=row_y_right,
            height=height,
            pad=pad,
            units=units,
            ax=ax,
            line_kws=line_kws,
            text_kws=text_kws,
        )
        results.append(artists)

        text_extent = _bbox_to_data(ax, artists.text.get_window_extent())
        min_next_y = _pixel_offset_to_data(ax, text_extent.top, offset=top_space, units=units)

    if min_next_y is not None and min_next_y > ax.get_ylim()[1]:
        ax.set_ylim(top=min_next_y)

    return results


# %%
if __name__ == '__main__':
    import numpy as np

    fig = plt.figure()
    ax = fig.gca()
    ax.boxplot(x=[np.linspace(1, 100), np.linspace(40, 140)], positions=[0, 1])
    ax.set_yscale('log', base=10)

    n_links = 10
    links(
        x_left=[0] * n_links,
        x_right=[1] * n_links,
        text=[f'test p-value = string {i}' for i in range(n_links)],
        y_left=[100] * n_links,
        y_right=[140] * n_links,
        ax=ax,
    )
    plt.show()
