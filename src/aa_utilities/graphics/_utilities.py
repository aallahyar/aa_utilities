"""Generic, matplotlib-oriented helpers for converting between pixel-based offsets/extents
and data coordinates. These are not specific to any single graphic element (e.g., links,
forest plots), so they can be reused wherever an annotation needs to reserve/measure space.
"""

from collections import namedtuple

from matplotlib import pyplot as plt, transforms as mpl_transforms

# a bounding box expressed in data coordinates
DataExtent = namedtuple('DataExtent', ['left', 'right', 'bottom', 'top'])


def _pixel_offset_to_data(ax, y_ref, offset, units='points'):
    """Converts a fixed on-screen offset (added above `y_ref`) into a data-y coordinate.

    `units='points'` is resolution-independent (a fixed physical size, e.g. 1/72 inch per
    point), so the same call produces a visually consistent gap regardless of the figure's
    DPI (screen vs. saved file). `units='dots'` is a raw pixel count instead, which will look
    like a different physical size at different DPIs.
    """
    if ax is None:
        ax = plt.gca()
    transform = mpl_transforms.offset_copy(ax.transData, y=offset, units=units, fig=ax.get_figure())
    return ax.transData.inverted().transform(transform.transform((0, y_ref)))[1]


def _bbox_to_data(ax, bbox):
    """Converts a display-space bbox (e.g., from `artist.get_window_extent()`) to a `DataExtent`."""
    (x0, y0), (x1, y1) = ax.transData.inverted().transform(bbox)
    return DataExtent(left=min(x0, x1), right=max(x0, x1), bottom=min(y0, y1), top=max(y0, y1))


def _measure_text_extent(ax, x, y, s, va='baseline', ha='center', transform=None, text_kws=None):
    """Measures the bounding box (in data coordinates) that a text string would occupy if
    drawn at `(x, y)`, without permanently adding it to the axes.

    A real `Text` artist is created (so real font metrics/styling, e.g. a `bbox` patch,
    rotation, or mathtext, are accounted for exactly), measured, then removed. Axes limits
    are left untouched.
    """
    if ax is None:
        ax = plt.gca()
    text_kws = text_kws or {}
    if transform is None:
        transform = ax.transData

    xlim, ylim = ax.get_xlim(), ax.get_ylim()  # guard against any unintended autoscale
    text_artist = ax.text(x, y, s, va=va, ha=ha, transform=transform, **text_kws)
    try:
        extent = _bbox_to_data(ax, text_artist.get_window_extent())
    finally:
        text_artist.remove()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    return extent
