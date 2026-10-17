"""Draws comparison "links" (a bracket connecting two positions with a text label next to
it, e.g. a p-value) on a matplotlib Axes, optionally stacking several of them so they don't
overlap. Works for vertical (x categories) and horizontal (y categories) plots.
"""

from dataclasses import dataclass, field

import numpy as np
from matplotlib import pyplot as plt, transforms as mpl_transforms

from ._utilities import _pixel_offset_to_data, _bbox_to_data, _to_native

_ORIENTATIONS = {'v': 'vertical', 'vertical': 'vertical', 'h': 'horizontal', 'horizontal': 'horizontal'}


@dataclass(frozen=True)
class LinkArtists:
    """One drawn link: its matplotlib artists plus where/how it was placed, so it can be
    identified and adjusted afterwards. "Value axis" is y for `orient='vertical'` and x for
    `orient='horizontal'`; all coordinates are in data units.

    Attributes:
        line: the `Line2D` of the bracket.
        text: the `Text` next to the bar.
        anchors: `(left, right)` categorical-axis positions the link connects, as given.
        level: 0-based stacking level (0 is the closest to the data).
        feet: `(left, right)` value-axis positions where the bracket's arms start.
        bar_value: value-axis position of the bracket's bar.
        bar_top: value-axis position reached by the rendered text plus `top_space`, i.e. what a
            higher-level, overlapping link has to clear.
    """

    line: object
    text: object
    anchors: tuple
    level: int
    feet: tuple
    bar_value: float
    bar_top: float


@dataclass
class LinksResult:
    """Result of `links()`. A plain object (not a tuple) so new fields can be added later
    without breaking existing attribute-based usage (`result.links`, `result.bases`, ...).

    Attributes:
        links: `list[LinkArtists]`, one per link, in input order.
        bases: per-position ceiling reached by anything touching that position.
        levels: stacking level of each link, in input order (same as `links[i].level`).
        orient: the normalized orientation, `'vertical'` or `'horizontal'`.
    """

    links: list = field(default_factory=list)
    bases: dict = field(default_factory=dict)
    levels: list = field(default_factory=list)
    orient: str = 'vertical'


def _normalize_orient(orient):
    if not isinstance(orient, str) or orient.lower() not in _ORIENTATIONS:
        raise ValueError(f"`orient` must be one of {sorted(_ORIENTATIONS)}. Got {orient!r}.")
    return _ORIENTATIONS[orient.lower()]


def _as_anchor_pairs(anchors):
    """Converts `anchors` (list of pairs, `(n, 2)` array, or 2-column DataFrame) into a list of
    `(left, right)` native-Python pairs."""
    try:
        values = anchors.to_numpy() if hasattr(anchors, 'to_numpy') else np.asarray(anchors)
    except ValueError as exc:  # ragged input
        raise ValueError('`anchors` must be an (n, 2) array-like of position pairs.') from exc
    if values.size == 0:
        return []
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(f'`anchors` must have shape (n, 2), one row per link. Got shape {values.shape}.')
    if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
        raise ValueError(f'`anchors` must be real numeric positions. Got dtype {values.dtype}.')
    if not np.isfinite(values).all():
        raise ValueError('`anchors` must not contain NaN or infinite values.')
    return [(_to_native(left), _to_native(right)) for left, right in values]


def _link(
    span_left,
    span_right,
    text,
    base_left,
    base_right=None,
    bar_value=None,
    height=10,
    pad=5,
    units='points',
    orient='vertical',
    ax=None,
    line_kws=None,
    text_kws=None,
):
    """Draws a single comparison bracket between `span_left` and `span_right`, with `text`
    placed next to it. Pure drawing primitive: never mutates `ax`'s limits.

    Parameters:
    ----------
    span_left, span_right: float
        positions along the categorical axis to link (e.g. x-tick indices for
        `orient='vertical'`, y-tick indices for `orient='horizontal'`).
    text: str
        text placed next to the link's bar.
    base_left, base_right: float
        positions, along the value axis, where the bracket's arms start (`base_right`
        defaults to `base_left`).
    bar_value: float, optional
        value-axis position of the bracket's bar. If not given, it is placed `height`
        `units` beyond `max(base_left, base_right)`.
    height: float
        extent of the bracket's arms, along the value axis, in `units`, when `bar_value` is
        not given.
    pad: float
        gap between `base_left`/`base_right`/`bar_value` and where the bracket is actually
        drawn, in `units`.
    units: str
        units for `height`/`pad` (and for `links()`'s `top_space`). `'points'` (default) is
        resolution-independent (a fixed physical size); `'dots'` is a fixed pixel count,
        which looks different physical sizes at different figure DPIs.
    orient: str
        `'vertical'`/`'v'` (default): the categorical axis is x, the value axis is y, and the
        bar grows upward. `'horizontal'`/`'h'`: the categorical axis is y, the value axis is
        x, and the bar grows rightward.
    ax: matplotlib Axes, optional
    line_kws, text_kws: dict, optional
        passed to `ax.plot()`/`ax.text()` respectively.

    Returns:
    -------
    (line, text, bar_value)
    """
    if ax is None:
        ax = plt.gca()
    if base_right is None:
        base_right = base_left
    orient = _normalize_orient(orient)
    is_vertical = orient == 'vertical'
    value_axis = 'y' if is_vertical else 'x'
    line_kws = {
        'linestyle': '-', 
        'color': '#aaaaaa', 
        'linewidth': 0.5,
    } | (line_kws or {}) # merge user-provided (higher priority) line_kws with defaults
    text_kws = {
        'color': '#aaaaaa',
        'fontsize': 10,
        } | (text_kws or {})

    if bar_value is None:
        bar_value = _pixel_offset_to_data(ax, max(base_left, base_right), offset=height, axis=value_axis, units=units)

    offset_kwargs = {'y': pad} if is_vertical else {'x': pad}
    transform_pad = mpl_transforms.offset_copy(ax.transData, units=units, fig=ax.get_figure(), **offset_kwargs)

    if is_vertical:
        plot_x = [span_left, span_left, span_right, span_right]
        plot_y = [base_left, bar_value, bar_value, base_right]
        text_x, text_y = (span_left + span_right) / 2, bar_value
        text_defaults = {'va': 'bottom', 'ha': 'center'}
    else:
        plot_x = [base_left, bar_value, bar_value, base_right]
        plot_y = [span_left, span_left, span_right, span_right]
        text_x, text_y = bar_value, (span_left + span_right) / 2
        text_defaults = {'va': 'center', 'ha': 'left'}

    line_artist = ax.plot(
        plot_x,
        plot_y,
        transform=transform_pad,
        **line_kws,
    )[0]
    text_artist = ax.text(
        text_x,
        text_y,
        text,
        transform=transform_pad,
        **(text_defaults | text_kws),
    )

    return line_artist, text_artist, bar_value


def links(
    anchors,
    texts,
    *,
    bases=None,
    colors=None,
    auto_order=True,
    height=10,
    pad=5,
    top_space=5,
    units='points',
    orient='vertical',
    ax=None,
    line_kws=None,
    text_kws=None,
):
    """Draws one or more comparison links, automatically stacking overlapping ones onto
    different levels so a link's bar always clears any position/link it visually spans over,
    and expands the value axis at most once to fit them all.

    Levels are assigned via a greedy interval-packing algorithm: two links share a level only
    if their spans don't overlap; a link's bar height then only needs to clear the real data
    in its own span plus any lower-level link whose span overlaps it - a position's true
    height (from `bases`) is never overwritten by an unrelated link merely passing over it,
    so the result doesn't depend on the order links are given in.

    Parameters:
    ----------
    anchors: array-like of shape (n, 2)
        one row per link with the two positions, along the categorical axis, that it connects
        (x-positions for `orient='vertical'`, y-positions for `orient='horizontal'`). A list
        of pairs, a numpy array or a 2-column DataFrame (column names are ignored). The order
        within a pair does not matter.
    texts: array-like of str
        text placed next to each link's bar, one entry per link.
    bases: Mapping or pd.Series, optional
        categorical position -> its real minimum extent along the value axis (e.g. a group's
        own data max). Any position not listed here defaults to the current axis boundary in
        the direction the brackets grow (`ax.get_ylim()[1]` for `'vertical'`,
        `ax.get_xlim()[1]` for `'horizontal'`).
    colors: array-like, optional
        per-link text color, one entry per link (e.g. to highlight significant comparisons).
        Applied to the text only (the bracket line stays as `line_kws` says), taking
        precedence over any shared `text_kws` color.
    auto_order: bool
        if True (default), links are packed into levels narrowest-span-first, which tends to
        minimize the number of levels used. If False, levels are assigned in the given input
        order instead - this never reintroduces the flattening issue above, but may use more
        levels (i.e. more stacking) than necessary.
    height: float
        extent of a link's arms, in `units`, from its feet up to its bar, along the value axis.
    pad: float
        gap between a link's nominal values and where it is actually drawn, in `units` -
        keeps the bracket from visually touching the data/position it starts from.
    top_space: float
        gap (in `units`) reserved past a link's rendered text before a higher-level,
        overlapping link (or the axis boundary) may start.
    units: str
        units for `height`/`pad`/`top_space`. `'points'` (default) is resolution-independent
        (a fixed physical size); `'dots'` is a fixed pixel count, which looks like a different
        physical size at different figure DPIs.
    orient: str
        `'vertical'`/`'v'` (default): `anchors` are x-positions, brackets grow upward and the
        y-axis is expanded to fit them - for vertical boxplots/bars. `'horizontal'`/`'h'`:
        `anchors` are y-positions, brackets grow rightward and the x-axis is expanded
        instead - for horizontal boxplots/bars.
    ax: matplotlib Axes, optional
    line_kws, text_kws: dict, optional
        passed to `ax.plot()`/`ax.text()` respectively, for styling the bracket/text.

    Returns:
    -------
    LinksResult(links, bases, levels, orient)
        `links`: list[LinkArtists], one per link, in input order. Each holds the `line` and
        `text` artists plus its `anchors`, `level`, `feet`, `bar_value` and `bar_top`, so any
        drawn element can be found and restyled/moved afterwards.
        `bases`: per-position ceiling reached by anything touching that position (its own
        `bases` value, or the top of the tallest link spanning over it, whichever is higher).
        `levels`: stacking level of each link, in input order.
        `orient`: the normalized orientation.
    """
    if ax is None:
        ax = plt.gca()
    orient = _normalize_orient(orient)
    is_vertical = orient == 'vertical'
    value_axis = 'y' if is_vertical else 'x'

    # normalize to native Python types so results (e.g. bases keys/values) don't leak
    # numpy scalars, regardless of whether the caller passed plain lists or numpy arrays
    anchor_pairs = _as_anchor_pairs(anchors)
    lefts = [left for left, _ in anchor_pairs]
    rights = [right for _, right in anchor_pairs]
    texts = list(texts)
    n = len(anchor_pairs)
    if len(texts) != n:
        raise ValueError(f'`texts` must have one entry per row of `anchors`. Got {len(texts)} vs {n}.')
    if colors is not None:
        colors = list(colors)
        if len(colors) != n:
            raise ValueError(f'`colors` must have one entry per row of `anchors`. Got {len(colors)} vs {n}.')
    if bases is None:
        bases = {}

    given_bases = {_to_native(k): _to_native(v) for k, v in bases.items()}
    default_base = _to_native(ax.get_ylim()[1] if is_vertical else ax.get_xlim()[1])
    positions = sorted(set(lefts) | set(rights) | set(given_bases))
    position_index = {p: i for i, p in enumerate(positions)}
    natural_base = {p: given_bases.get(p, default_base) for p in positions}

    spans = [tuple(sorted((position_index[lefts[i]], position_index[rights[i]]))) for i in range(n)]


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
        foot_left = ceiling(level, position_index[lefts[i]])
        foot_right = ceiling(level, position_index[rights[i]])

        link_text_kws = text_kws
        if colors is not None:
            link_text_kws = {**(text_kws or {}), 'color': colors[i]}

        line_artist, text_artist, bar_value = _link(
            span_left=lefts[i],
            span_right=rights[i],
            text=texts[i],
            base_left=foot_left,
            base_right=foot_right,
            bar_value=_pixel_offset_to_data(ax, clearance_ref, offset=height, axis=value_axis, units=units),
            pad=pad,
            units=units,
            orient=orient,
            ax=ax,
            line_kws=line_kws,
            text_kws=link_text_kws,
        )

        text_extent = _bbox_to_data(ax, text_artist.get_window_extent())
        bar_edge = text_extent.top if is_vertical else text_extent.right
        bar_top = _pixel_offset_to_data(ax, bar_edge, offset=top_space, axis=value_axis, units=units)
        results[i] = LinkArtists(
            line=line_artist,
            text=text_artist,
            anchors=(lefts[i], rights[i]),
            level=level,
            feet=(foot_left, foot_right),
            bar_value=bar_value,
            bar_top=bar_top,
        )
        covered.append((level, spans[i], bar_top))
        for p in positions[lo : hi + 1]:
            final_heights[p] = max(final_heights[p], bar_top)
        if overall_top is None or bar_top > overall_top:
            overall_top = bar_top
            top_text_artist = text_artist


    def _value_limit():
        return ax.get_ylim()[1] if is_vertical else ax.get_xlim()[1]

    def _set_value_limit(value):
        if is_vertical:
            ax.set_ylim(top=value)
        else:
            ax.set_xlim(right=value)

    if overall_top is not None:
        _set_value_limit(max(overall_top, _value_limit()))

        # changing the value-axis limit (especially on a log scale) shifts the data<->pixel
        # mapping used above, so the topmost text's clearance must be re-checked (and
        # re-grown if needed) against the *new* scale - otherwise it can end up poking past
        # the new axis boundary.
        for _ in range(4):
            text_extent = _bbox_to_data(ax, top_text_artist.get_window_extent())
            bar_edge = text_extent.top if is_vertical else text_extent.right
            required_top = _pixel_offset_to_data(ax, bar_edge, offset=top_space, axis=value_axis, units=units)
            if required_top <= _value_limit():
                break
            _set_value_limit(required_top)

    return LinksResult(
        links=results,
        bases=final_heights,
        levels=[level_of[i] for i in range(n)],
        orient=orient,
    )




