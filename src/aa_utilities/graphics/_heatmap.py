import textwrap
from functools import cache

import numpy as np
from matplotlib import colors as mpl_colors
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from matplotlib.markers import MarkerStyle
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib.transforms import Affine2D
from seaborn.matrix import ClusterGrid


def _find_mesh(ax):
    """Returns the heatmap's QuadMesh drawn on `ax`."""
    for collection in ax.collections:
        if isinstance(collection, QuadMesh):
            return collection
    raise ValueError('No heatmap found in `ax`; draw one with `sns.heatmap()` first.')


def _resolve_target(target):
    """Returns (heatmap axes, data shape, row order, col order) for a seaborn clustermap or heatmap axes.

    The orders map the original data order to the drawn order (the identity for a plain heatmap).
    """
    if isinstance(target, ClusterGrid):
        ax, shape = target.ax_heatmap, target.data2d.shape  # `data2d` is already in the clustered (drawn) layout
        row_order = list(range(shape[0])) if target.dendrogram_row is None else target.dendrogram_row.reordered_ind
        col_order = list(range(shape[1])) if target.dendrogram_col is None else target.dendrogram_col.reordered_ind
        return ax, shape, row_order, col_order
    if isinstance(target, Axes):
        shape = _find_mesh(target).get_array().shape
        return target, shape, list(range(shape[0])), list(range(shape[1]))
    raise TypeError(f'`target` must be a seaborn ClusterGrid or the Axes of `sns.heatmap()`, got {type(target).__name__}')


def _default_legend_rect(ax):
    """Inset rectangle (in axes fractions) just outside the heatmap, past a colorbar sitting on its right."""
    left = 1.05
    colorbar = getattr(_find_mesh(ax), 'colorbar', None)
    if colorbar is not None:
        ax_box, colorbar_box = ax.get_position(), colorbar.ax.get_position()
        if colorbar_box.x0 >= ax_box.x1:
            left = (colorbar_box.x1 - ax_box.x0) / ax_box.width + 0.05
    return [left, 0, 0.15, 0.3]


def _rgba2hex(rgba_arr):
    return np.apply_along_axis(
        mpl_colors.to_hex, 
        axis=1, 
        arr=rgba_arr, 
        keep_alpha=True,
    )


def _to_grid(value, shape, name, is_color=False):
    """Broadcast a scalar (or, for colors, a single color) to `shape`, or validate an array that already has it.

    Colors may also be arrays of shape `shape` or `shape + (n_channels,)`.
    """
    if is_color:
        if mpl_colors.is_color_like(value):
            if isinstance(value, str):
                return np.full(shape, value)
            return np.tile(mpl_colors.to_rgba(value), (*shape, 1))  # exact RGBA, no 8-bit rounding of a hex string
        array = np.asarray(value)
        if array.ndim in (2, 3) and array.shape[:2] == shape:
            return array
        expected = f'a single color or an array of shape {shape} (or {shape + (4,)})'
    else:
        try:
            array = np.asarray(value, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'`{name}` must be numeric') from exc
        if array.ndim == 0:
            return np.full(shape, array)
        if array.shape == shape:
            return array
        expected = f'a scalar or an array of shape {shape}'
    raise ValueError(f'`{name}` must be {expected}, got {np.shape(value)}')


# Filled matplotlib marker codes, plus '?' (drawn by `_question_mark_path`); unfilled markers ('+', 'x', '_', ...)
# have no area to outline.
_SHAPES = ('s', 'o', 'D', 'd', '^', 'v', '<', '>', 'p', 'h', 'H', '8', '*', 'P', 'X', '?')
_DEFAULT_SHAPE = 's'


def _to_shape_grid(value, shape):
    """Broadcast a single shape code to `shape`, or validate an array of codes that already has it."""
    array = np.asarray(_DEFAULT_SHAPE if value is None else value, dtype=object)
    if array.ndim == 0:
        array = np.full(shape, array.item(), dtype=object)
    elif array.shape != shape:
        raise ValueError(f'`shapes` must be a single code or an array of shape {shape}, got {array.shape}')
    unknown = {code for code in array.ravel().tolist() if not (isinstance(code, str) and code in _SHAPES)}
    if unknown:
        raise ValueError(
            f'`shapes` has unsupported code(s) {sorted(map(repr, unknown))}; supported: {", ".join(_SHAPES)}. '
            "Use 'P' and 'X' for a plus and a cross."
        )
    return array


def _question_mark_path(thickness=0.24, width=0.7, hook_radius_y=0.2, dot_gap=0.08):
    """Closed outline of a '?' with strokes of constant `thickness`, as tall as the centred unit square and `width` wide."""
    half = thickness / 2
    radius_x = (width - thickness) / 2  # the hook's outer edge touches the left and right sides of the `width`
    center_y = 0.5 - half - hook_radius_y  # ... and its top side touches the top of the square

    # centreline: an elliptical hook from its left tip over the top and down the right side, a quarter-circle-like
    # turn into the vertical stem, and the stem down to a gap above the dot. The hook stops early enough for that
    # turn to be wider than half the thickness, otherwise the inner edge of the stroke folds over itself.
    end_angle = -np.degrees(np.arccos(min(1, 1.1 * half / radius_x)))
    angles = np.radians(np.arange(200, end_angle - 1, -2))
    hook = np.column_stack([radius_x * np.cos(angles), center_y + hook_radius_y * np.sin(angles)])
    start, direction = hook[-1], hook[-1] - hook[-2]
    direction /= np.linalg.norm(direction)
    turn = start[0]
    stem_top = np.array([0, start[1] - turn])
    stem_end = np.array([0, -0.5 + thickness + dot_gap])
    handles = [start + 0.55 * turn * direction, stem_top + [0, 0.55 * turn]]
    t = np.linspace(0, 1, 40)[1:, None]
    curve = (1 - t) ** 3 * start + 3 * (1 - t) ** 2 * t * handles[0] + 3 * (1 - t) * t**2 * handles[1] + t**3 * stem_top
    stem = stem_top + np.linspace(0, 1, 10)[1:, None] * (stem_end - stem_top)
    centerline = np.vstack([hook, curve, stem])

    # offset the centreline to both sides by half the thickness, and join them into one polygon
    tangent = np.gradient(centerline, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    outline = np.vstack([centerline + half * normal, (centerline - half * normal)[::-1]])

    dot = Path.circle((0, -0.5 + half), half)
    return Path.make_compound_path(Path(np.vstack([outline, outline[:1]]), closed=True), dot)


@cache
def _unit_shape(code):
    """Outline of marker `code`, centred on the origin and scaled so its larger bounding-box side is 1."""
    if code == '?':
        path = _question_mark_path()
    else:
        marker = MarkerStyle(code)
        path = marker.get_path().transformed(marker.get_transform())
    box = path.get_extents()
    fit = Affine2D().translate(-(box.x0 + box.x1) / 2, -(box.y0 + box.y1) / 2).scale(1 / max(box.width, box.height))
    return fit.transform_path(path)


def _shape_patch(ax, code, center_x, center_y, size, **style):
    """Patch of shape `code` fitted to the `size` x `size` square centred on (center_x, center_y)."""
    # a heatmap's y axis is inverted: flip the outline there, so that '^' still points up on screen
    flip_x = -1 if ax.xaxis_inverted() else 1
    flip_y = -1 if ax.yaxis_inverted() else 1
    placed = Affine2D().scale(flip_x * size, flip_y * size).translate(center_x, center_y)
    return PathPatch(placed.transform_path(_unit_shape(code)), **style)


def _reference_shape(shapes):
    """Shape drawn by the size legend: the one shared by every cell, otherwise a square."""
    unique = np.unique(shapes)
    return unique[0] if len(unique) == 1 else _DEFAULT_SHAPE


def _resolve_legend_bins(sizes, bins):
    if np.ndim(bins) == 0:
        if not isinstance(bins, (int, np.integer)):
            raise ValueError('legend `bins` must be a positive integer or a non-empty array-like of numeric values')
        if bins <= 0:
            raise ValueError('legend `bins` must be a positive integer')
        return np.linspace(sizes.min(), sizes.max(), int(bins))

    bin_values = np.asarray(bins)
    if bin_values.size == 0:
        raise ValueError('legend `bins` must be a non-empty array-like of numeric values')

    try:
        return bin_values.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError('legend `bins` must contain only numeric values') from exc


def _extract_face_colors(ax):
    """Extract per-cell RGBA colors from the seaborn heatmap QuadMesh."""

    quad_mesh = _find_mesh(ax)
    n_rows, n_cols = quad_mesh.get_array().shape

    # Force a canvas draw so that QuadMesh expands broadcast colors to per-cell
    # values before we read them.  Without this, a second call (e.g. two
    # consecutive overlay_shapes() calls) returns a single RGBA row instead of
    # one row per cell.
    ax.get_figure().canvas.draw()

    # Prefer colors stashed before any _overlay_shapes() dimming; otherwise a
    # later call would extract this mesh's already-dimmed (background_alpha)
    # state instead of the true original colors.
    face_colors = getattr(quad_mesh, '_facecolors_pristine', None)
    if face_colors is None:
        face_colors = quad_mesh.get_facecolors()
        if face_colors is None or len(face_colors) == 0:
            face_colors = quad_mesh._facecolors

    face_colors_hex = np.apply_along_axis(
        mpl_colors.to_hex, 
        axis=1, 
        arr=face_colors, 
        keep_alpha=True,
    )

    return face_colors_hex.reshape(n_rows, n_cols)


def _overlay_shapes(ax, shape, face_colors, sizes, shape_kws):
    """Draw sized shapes (squares by default) on top of the heatmap mesh (skipping hidden cells) and dim the original."""
    from matplotlib.collections import PatchCollection

    shapes = _to_shape_grid(shape_kws.get('shapes'), shape)
    edgecolors = shape_kws.get('edgecolors')
    edgecolors = _to_grid('none' if edgecolors is None else edgecolors, shape, 'edgecolors', is_color=True)
    linewidths = shape_kws.get('linewidths')
    linewidths = _to_grid(1.5 if linewidths is None else linewidths, shape, 'linewidths')

    # Dim the original heatmap mesh by setting its alpha to the specified background_alpha.
    background_alpha = shape_kws.get('background_alpha', 0.1)
    quad_mesh = _find_mesh(ax)
    is_hidden = np.ma.getmaskarray(quad_mesh.get_array())  # cells hidden by seaborn (`mask=` or NaN)
    # Stash the pristine (pre-dimming) colors once, so a later call's automatic
    # face-color extraction (facecolors=None) doesn't inherit this dimming.
    if not hasattr(quad_mesh, '_facecolors_pristine'):
        ax.get_figure().canvas.draw()
        quad_mesh._facecolors_pristine = quad_mesh.get_facecolors().copy()
    quad_mesh.set_alpha(background_alpha)

    boxes = []
    for ri in range(shape[0]):
        for ci in range(shape[1]):
            if is_hidden[ri, ci]:
                continue
            boxes.append(
                _shape_patch(
                    ax,
                    shapes[ri, ci],
                    ci + 0.5,
                    ri + 0.5,
                    sizes[ri, ci],
                    facecolor=face_colors[ri, ci],
                    edgecolor=edgecolors[ri, ci],
                    linewidth=linewidths[ri, ci],
                )
            )
    patch_collection = PatchCollection(boxes, match_original=True)
    ax.add_collection(patch_collection)

    return patch_collection


def _draw_size_legend(ax, ax_heat, sizes, legend_kws, shape=_DEFAULT_SHAPE):
    """Draw a size legend showing representative shape sizes.

    The legend axes xlim/ylim are derived from the *rendered* pixel size
    of ``ax`` and ``ax_heat`` (via ``get_window_extent``), so that 1
    data-unit maps to the same physical (pixel) size on both, regardless
    of how ``ax`` was created (GridSpec cell, ``inset_axes``, a
    user-supplied axes anywhere in the figure, ...) and even if
    ``ax_heat``'s plotted area doesn't fill its full allocated box (e.g.
    an equal-aspect heatmap that got letterboxed). A v×v rectangle drawn
    here is pixel-identical to one on the heatmap.

    Args:
        ax: Axes to draw the legend into.
        ax_heat: The heatmap axes to match scale against.
        sizes (np.ndarray): The per-cell sizes matrix (typically in [0, 1]).
        legend_kws (dict): Configuration dict. Keys:
            - shape (str): Marker code of the legend boxes. Default: `shape`.
            - bins (int or array-like): Number of evenly spaced sizes or
              explicit values. Default: 4.
            - labels (list[str]): Explicit labels per bin.
            - label_fmt (str): Format string for auto labels. Default: '{:.2f}'.
            - title (str): Legend title. Default: None.
            - fontsize (int): Label font size. Default: 8.
            - title_fontsize (int): Title font size. Default: 9.
            - facecolor (str): Box fill color. Default: '#cccccc'.
            - edgecolor (str): Box border color. Default: '#333333'.
            - linewidth (float): Box border width. Default: 1.
        shape (str): Marker code of the legend boxes, unless `legend_kws` has a ``'shape'`` key.
    """
    shape_code = _to_shape_grid(legend_kws.get('shape', shape), (1, 1))[0, 0]

    bins = legend_kws.get('bins', 4)
    bin_values = _resolve_legend_bins(sizes, bins)
    n_bins = len(bin_values)

    labels = legend_kws.get('labels', None)
    if labels is None:
        label_fmt = legend_kws.get('label_fmt', '{:.2f}')
        labels = [label_fmt.format(v) for v in bin_values]
    else:
        if 'label_fmt' in legend_kws:
            raise ValueError('Cannot specify both `labels` and `label_fmt` in legend_kws')
    if len(labels) != n_bins:
        raise ValueError(f'Number of labels ({len(labels)}) must match number of bins ({n_bins})')

    facecolor = legend_kws.get('facecolor', '#cccccc')
    edgecolor = legend_kws.get('edgecolor', '#333333')
    linewidth = legend_kws.get('linewidth', 1)
    fontsize = legend_kws.get('fontsize', 8)
    title = legend_kws.get('title', None)
    title_fontsize = legend_kws.get('title_fontsize', 9)

    # Match data-to-pixel scale with the heatmap so v×v boxes look identical.
    # Uses actual rendered pixel size (not GridSpec ratios) so this works
    # regardless of how `ax` was positioned relative to `ax_heat`.
    fig = ax.get_figure()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    heat_bbox = ax_heat.get_window_extent(renderer)
    legend_bbox = ax.get_window_extent(renderer)

    heat_xrange = np.abs(np.diff(ax_heat.get_xlim()))[0]
    heat_yrange = np.abs(np.diff(ax_heat.get_ylim()))[0]
    pixels_per_unit_x = heat_bbox.width / heat_xrange
    pixels_per_unit_y = heat_bbox.height / heat_yrange

    marker_xmax = legend_bbox.width / pixels_per_unit_x
    marker_ymax = legend_bbox.height / pixels_per_unit_y

    ax.set_xlim(0, marker_xmax)
    ax.set_ylim(marker_ymax, 0)
    ax.set_xticks([])
    ax.set_yticks([])

    # Spacing: largest marker + 10% gap, constant across all bins.
    # If bins don't fit, shrink spacing so all markers stay within ylim.
    max_v = max(bin_values)
    ideal_spacing = max_v + max_v / 10  # marker height + 10%-marker gap
    spacing = min(ideal_spacing, marker_ymax / n_bins)

    # x-coordinate of the right edge shared by all boxes (right-aligns all sizes).
    # `max_v` (not `max_v / 2`) so the largest box's left edge sits at 0, flush with the axes'
    # own left edge instead of straddling it and bleeding into whatever is to the left (e.g. the
    # heatmap itself).
    right_edge = max_v
    for mi, (v, lbl) in enumerate(reversed(list(zip(bin_values, labels)))):
        cy = mi * spacing + spacing / 2 + spacing / 10  # center y of the current marker
        ax.add_patch(
            _shape_patch(
                ax,
                shape_code,
                right_edge - v / 2,
                cy,
                v,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=linewidth,
                clip_on=False,
            )
        )
        ax.text(
            right_edge + max_v * 0.1,  # label starts just past the largest box
            cy,
            textwrap.fill(lbl, width=10),
            va='center_baseline',
            ha='left',
            fontsize=fontsize,
        )

    if title:
        ax.set_title(
            title,
            va='center',
            ha='center',
            fontsize=title_fontsize,
            fontweight='normal',
        )

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.patch.set_visible(False)


def overlay_shapes(
    target,
    sizes=None,
    facecolors=None,
    edgecolors=None,
    linewidths=None,
    background_alpha=0.1,
    legend=None,
    shapes=None,
):
    """Overlay sized shapes (squares by default) on a seaborn clustermap or heatmap.

    The per-cell options (`sizes`, `facecolors`, `edgecolors`, `linewidths`, `shapes`) accept either a full
    matrix in the **original** data order (pre-clustering for a clustermap), or a single value
    applied to every cell. Cells hidden by seaborn (`mask=` or NaN) get no shape.

    Args:
        target: The ``ClusterGrid`` returned by ``sns.clustermap()``, or the ``Axes`` returned by
            ``sns.heatmap()``.
        sizes (float or np.ndarray, optional): Per-cell shape sizes. 1 fills a cell; values in [0, 1]
            are recommended, but larger ones are allowed (shapes then overlap their neighbours).
            Default: 0.8 uniform.
        facecolors (color or np.ndarray, optional): A single color, or per-cell RGBA colors with
            shape either `(n_rows, n_cols, 4)` or `(n_rows, n_cols)`.  If not provided, colors are
            extracted from the heatmap mesh.  This is necessary if the heatmap was drawn with a
            custom colormap.
        edgecolors (color or np.ndarray, optional): A single color, or per-cell edge colors in any
            valid matplotlib format (e.g., '#RRGGBB', (r, g, b), etc.).
            Default: no (i.e., fully transparent) edge color.
        linewidths (float or np.ndarray, optional): Per-cell border widths.
            Default: 1.5 uniform.
        background_alpha (float): Heatmap mesh alpha after dimming.
            Default: 0.1.
        legend (dict, optional): Size legend config, passed to
            ``_draw_size_legend`` (see that function for keys such as
            ``bins``, ``labels``, ``title``, ...). If ``None`` (default),
            no legend is drawn. If a dict is given without an ``'ax'``
            key, a narrow inset axes is created just outside the
            heatmap's bottom-right corner (past the colorbar, if that sits on the right);
            pass ``{'ax': my_ax}`` to draw into an axes of your own instead (see the example).
            The legend shapes use the shape shared by all cells, or squares if the shapes are mixed;
            add ``'shape': <code>`` to choose another one.
        shapes (str or np.ndarray, optional): A single matplotlib marker code applied to every
            cell, or per-cell codes. Supported: 's' (square), 'o' (circle), 'D' and 'd' (diamonds),
            '^', 'v', '<' and '>' (triangles), 'p' (pentagon), 'h' and 'H' (hexagons), '8' (octagon),
            '*' (star), 'P' (plus), 'X' (cross) and '?' (question mark). Unfilled markers such as '+' and
            'x' are not supported. Every shape is scaled so that its larger bounding-box side equals `size`
            (a `size` of 1 reaches the cell edges in that direction); shapes that are narrower than tall
            ('d', 'h', 'H', '?') fill less of the other, and the shapes differ in how much of the square
            they cover. Default: 's'.

    Returns:
        matplotlib.collections.PatchCollection: The overlay patch collection.

    Notes:
        When ``row_cluster=False`` or ``col_cluster=False`` was passed to
        ``sns.clustermap()``, the corresponding dendrogram is ``None`` and
        the original row/column order is used as-is.

    Example:
        A heatmap with its own colorbar and size-legend axes, laid out with a ``GridSpec``::

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import seaborn as sns
            from aa_utilities.graphics import overlay_shapes

            values = np.linspace(-1, 1, 64).reshape(8, 8)
            df = pd.DataFrame(values)

            fig = plt.figure(figsize=(8, 6))
            grid = fig.add_gridspec(2, 2, width_ratios=[10, 0.5], height_ratios=[1, 1], hspace=0.5, wspace=0.1)
            heat_ax = fig.add_subplot(grid[:, 0])    # large axes: the heatmap
            cbar_ax = fig.add_subplot(grid[0, 1])    # colorbar
            legend_ax = fig.add_subplot(grid[1, 1])  # size legend

            sns.heatmap(df, ax=heat_ax, cbar_ax=cbar_ax, cmap='vlag')
            overlay_shapes(
                heat_ax,
                sizes=np.abs(values),                    # the size encodes |value|
                shapes=np.where(values >= 0, '^', 'v'),  # the shape encodes the sign
                background_alpha=0.5,
                legend={'ax': legend_ax, 'bins': [0.25, 0.5, 1.0], 'title': '|value|'},
            )
            plt.show()
    """
    heat_ax, shape, row_order, col_order = _resolve_target(target)

    # Reorder relevant data from original data order to the visual (clustered) order.
    sizes = _to_grid(0.8 if sizes is None else sizes, shape, 'sizes')
    sizes_reordered = sizes[np.ix_(row_order, col_order)]

    if facecolors is None:
        facecolors = _extract_face_colors(heat_ax)
        # Already in clustered (visual) order — no reordering needed.
        facecolors_reordered = facecolors
    else:
        # Reorder from original data order to the visual (clustered) order.
        facecolors = _to_grid(facecolors, shape, 'facecolors', is_color=True)
        facecolors_reordered = facecolors[np.ix_(row_order, col_order)] # alternative to `facecolors[row_order, :][:, col_order]` approach

    edgecolors = _to_grid('none' if edgecolors is None else edgecolors, shape, 'edgecolors', is_color=True)
    edgecolors_reordered = edgecolors[np.ix_(row_order, col_order)]

    linewidths = _to_grid(1.5 if linewidths is None else linewidths, shape, 'linewidths')
    linewidths_reordered = linewidths[np.ix_(row_order, col_order)]

    shapes = _to_shape_grid(shapes, shape)
    shapes_reordered = shapes[np.ix_(row_order, col_order)]

    shape_kws = {
        'edgecolors': edgecolors_reordered,
        'linewidths': linewidths_reordered,
        'shapes': shapes_reordered,
        'background_alpha': background_alpha,
    }

    # Draw the shapes on top of the heatmap mesh
    patch_collection = _overlay_shapes(
        ax=heat_ax,
        shape=shape,
        face_colors=facecolors_reordered,
        sizes=sizes_reordered,
        shape_kws=shape_kws,
    )

    if legend is not None:
        legend_kws = dict(legend)
        ax_legend = legend_kws.get('ax')
        if ax_legend is None:
            ax_legend = heat_ax.inset_axes(_default_legend_rect(heat_ax))
        ax_legend.set_label('size_legend')
        _draw_size_legend(ax_legend, heat_ax, sizes_reordered, legend_kws, shape=_reference_shape(shapes))

    return patch_collection
