import textwrap

import numpy as np
import seaborn as sns
from matplotlib import (
    pyplot as plt,
    colors as mpl_colors,
)


from ..loggers import setup_logger

logger = setup_logger('Heatmap', level='DEBUG')


def _rgba2hex(rgba_arr):
    return np.apply_along_axis(
        mpl_colors.to_hex, 
        axis=1, 
        arr=rgba_arr, 
        keep_alpha=True,
    )


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

    quad_mesh = ax.collections[0]
    n_rows, n_cols = quad_mesh.get_array().shape

    # Force a canvas draw so that QuadMesh expands broadcast colors to per-cell
    # values before we read them.  Without this, a second call (e.g. two
    # consecutive overlay_boxes() calls) returns a single RGBA row instead of
    # one row per cell.
    ax.get_figure().canvas.draw()

    # Prefer colors stashed before any _overlay_boxes() dimming; otherwise a
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


def _overlay_boxes(ax, heatmap_df, face_colors, sizes, box_kws):
    """Draw sized rectangles on top of the heatmap mesh and dim the original."""
    from matplotlib import patches
    from matplotlib.collections import PatchCollection

    edgecolors = box_kws.get('edgecolors')
    if edgecolors is None:
        edgecolors = np.full(heatmap_df.shape, fill_value='none')  # no edge color
    linewidths = box_kws.get('linewidths')
    if linewidths is None:
        linewidths = np.ones(heatmap_df.shape, dtype=float) * 1.5

    # Dim the original heatmap mesh by setting its alpha to the specified background_alpha.
    background_alpha = box_kws.get('background_alpha', 0.1)
    quad_mesh = ax.collections[0]
    # Stash the pristine (pre-dimming) colors once, so a later call's automatic
    # face-color extraction (facecolors=None) doesn't inherit this dimming.
    if not hasattr(quad_mesh, '_facecolors_pristine'):
        ax.get_figure().canvas.draw()
        quad_mesh._facecolors_pristine = quad_mesh.get_facecolors().copy()
    quad_mesh.set_alpha(background_alpha)

    rectangles = []
    for ri in range(heatmap_df.shape[0]):
        for ci in range(heatmap_df.shape[1]):
            rectangles.append(
                patches.Rectangle(
                    (ci + 0.5 - sizes[ri, ci] / 2, ri + 0.5 - sizes[ri, ci] / 2),
                    width=sizes[ri, ci],
                    height=sizes[ri, ci],
                    facecolor=face_colors[ri, ci],
                    edgecolor=edgecolors[ri, ci],
                    linewidth=linewidths[ri, ci],
                )
            )
    patch_collection = PatchCollection(rectangles, match_original=True)
    ax.add_collection(patch_collection)

    return patch_collection


def _draw_box_legend(ax, ax_heat, sizes, legend_kws):
    """Draw a marker-range legend showing representative box sizes.

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
        sizes (np.ndarray): The per-cell sizes matrix (values in [0, 1]).
        legend_kws (dict): Configuration dict. Keys:
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
    """
    from matplotlib import patches

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
    right_edge = max_v / 2
    for mi, (v, lbl) in enumerate(reversed(list(zip(bin_values, labels)))):
        cy = mi * spacing + spacing / 2 + spacing / 10  # center y of the current marker
        ax.add_patch(
            patches.Rectangle(
                (right_edge - v, cy - v / 2),
                width=v,
                height=v,
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


def heatmap(matrix_df, box_kws, fig=None, gs_kws=None, **heat_kws):
    """Plot a heatmap with sized rectangles per cell and a marker-range legend.

    Creates a figure with a 2x2 GridSpec::

        +----------------+-----------+
        |                | colorbar  |
        |    heatmap     +-----------+
        |                | markers   |
        +----------------+-----------+

    Access individual axes via ``fig.axes`` (creation order: heatmap,
    colorbar, marker_legend) or by label, e.g.::

        next(a for a in fig.axes if a.get_label() == 'heatmap')

    Args:
        matrix_df (pd.DataFrame): Data matrix.
        box_kws (dict): Rectangle overlay configuration:
            - sizes (np.ndarray): Per-cell box sizes in [0, 1].
              Default: 0.8 uniform.
            - edgecolors (np.ndarray): Per-cell edge colors. Default: None.
            - linewidths (np.ndarray): Per-cell border widths. Default: 1.5 uniform.
            - background_alpha (float): Original heatmap (mesh) alpha. Default: 0.1.
            - legend (dict): Marker legend config passed to
                            ``_draw_box_legend``. Omit the key or pass ``{}`` to draw the
                            default 4-bin auto legend. Pass ``None`` to suppress the legend.
        fig (matplotlib.figure.Figure, optional): Pre-created figure to draw
            into.  If ``None``, a new ``(10, 8)`` figure is created.
        gs_kws (dict): Keyword arguments forwarded to
            ``fig.add_gridspec()``.  Use this to control the layout, e.g.
            ``{'width_ratios': [10, 0.5], 'height_ratios': [1, 1],
            'wspace': 0.3, 'hspace': 0.5}``.
        **heat_kws: Forwarded to ``sns.heatmap()``.  The ``ax`` key is ignored;
            ``cbar_ax`` is overridden to point at the colorbar subplot.

    Returns:
        matplotlib.figure.Figure

    
    Example: See ``dev_scripts/debugging_scripts/graphics/heatmap_example.py`` for usage examples.
    """

    gs_kws = dict(gs_kws or {})
    gs_kws.setdefault('width_ratios', [10, 0.5])
    gs_kws.setdefault('height_ratios', [1, 1])
    gs_kws.setdefault('hspace', 0.5)
    gs_kws.setdefault('wspace', 0.1)

    if fig is None:
        fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 2, **gs_kws)

    heat_ax = fig.add_subplot(gs[:, 0])
    cbar_ax = fig.add_subplot(gs[0, 1])
    marker_ax = fig.add_subplot(gs[1, 1])

    heat_ax.set_label('heatmap')
    cbar_ax.set_label('colorbar')
    marker_ax.set_label('marker_legend')

    # Redirect colorbar to its dedicated axes; ignore user-supplied ax
    if 'ax' in heat_kws or 'cbar_ax' in heat_kws:
        logger.warning(
            '`ax` and `cbar_ax` arguments are ignored; this function produces its own heatmap, '
            'colorbar, and marker-legend axes. You may provide a pre-created figure'
            ' via the `fig` argument, or adjust the layout via `gs_kws`.'
        )
        heat_kws.pop('ax', None)
        heat_kws.pop('cbar_ax', None)

    # Produce the heatmap
    sns.heatmap(data=matrix_df, ax=heat_ax, cbar_ax=cbar_ax, **heat_kws)

    # Overlay boxes
    n_rows, n_cols = matrix_df.shape
    face_colors = _extract_face_colors(heat_ax)
    sizes = box_kws.get('sizes', np.ones((n_rows, n_cols)) * 0.8)
    _overlay_boxes(heat_ax, matrix_df, face_colors, sizes, box_kws)

    # Draw marker legend
    legend_kws = box_kws.get('legend', {})
    if legend_kws is not None:
        _draw_box_legend(marker_ax, heat_ax, sizes, legend_kws)

    return fig


def overlay_boxes(
    clustermap_obj,
    sizes=None,
    facecolors=None,
    edgecolors=None,
    linewidths=None,
    background_alpha=0.1,
    legend=None,
):
    """Overlay sized rectangles on an existing seaborn clustermap's heatmap.

    Args:
        clustermap_obj: The ``ClusterGrid`` object returned by
            ``sns.clustermap()``.
        sizes (np.ndarray, optional): Per-cell box sizes in [0, 1], in
            **original** data order (pre-clustering).
            Default: 0.8 uniform.
        facecolors (np.ndarray, optional): Per-cell RGBA colors in the original
            data order.  If not provided, colors are extracted from the heatmap
            mesh.  This is necessary if the heatmap was drawn with a custom colormap.
        edgecolors (np.ndarray, optional): Per-cell edge colors in any valid
            matplotlib format (e.g., '#RRGGBB', (r, g, b), etc.), in **original**
            data order.
            Default: no (i.e., fully transparent) edge color.
        linewidths (np.ndarray, optional): Per-cell border widths in **original** data order.
            Default: 1.5 uniform.
        background_alpha (float): Heatmap mesh alpha after dimming.
            Default: 0.1.
        legend (dict, optional): Marker legend config, passed to
            ``_draw_box_legend`` (see that function for keys such as
            ``bins``, ``labels``, ``title``, ...). If ``None`` (default),
            no legend is drawn. If a dict is given without an ``'ax'``
            key, a narrow inset axes is created just outside the
            heatmap's bottom-right corner; pass ``{'ax': my_ax}`` to draw
            into an axes of your own instead.

    Returns:
        matplotlib.collections.PatchCollection: The overlay patch collection.

    Notes:
        When ``row_cluster=False`` or ``col_cluster=False`` was passed to
        ``sns.clustermap()``, the corresponding dendrogram is ``None`` and
        the original row/column order is used as-is.
    """
    heat_ax = clustermap_obj.ax_heatmap

    # .data is the original DataFrame passed to clustermap()
    # .data2d has already been reordered to the clustered (visual) layout by seaborn 
    # and matches the heatmap's rendered/drawn order.
    heat_df = clustermap_obj.data2d
    n_rows, n_cols = heat_df.shape

    # Resolve row/col clustering order, falling back to identity when a
    # dendrogram is absent (row_cluster=False or col_cluster=False).
    if clustermap_obj.dendrogram_row is not None:
        row_order = clustermap_obj.dendrogram_row.reordered_ind
    else:
        row_order = list(range(n_rows))

    if clustermap_obj.dendrogram_col is not None:
        col_order = clustermap_obj.dendrogram_col.reordered_ind
    else:
        col_order = list(range(n_cols))

    # Reorder relevant data from original data order to the visual (clustered) order.
    if sizes is None:
        sizes = np.ones((n_rows, n_cols)) * 0.8
    sizes = np.array(sizes)  # ensure it's a numpy array for indexing
    sizes_reordered = sizes[np.ix_(row_order, col_order)]

    if facecolors is None:
        facecolors = _extract_face_colors(heat_ax)
        # Already in clustered (visual) order — no reordering needed.
        facecolors_reordered = facecolors
    else:
        # Reorder from original data order to the visual (clustered) order.
        facecolors = np.array(facecolors)
        facecolors_reordered = facecolors[row_order, :][:, col_order] # alternative to `np.ix_()` approach

    if edgecolors is None:
        # default to no edge color
        edgecolors = np.full((n_rows, n_cols), fill_value='none') # no edge color
    edgecolors = np.asarray(edgecolors)  # ensure it's a numpy array for indexing
    # Reorder edgecolors from original data order to the visual (clustered) order.
    edgecolors_reordered = edgecolors[np.ix_(row_order, col_order)]

    if linewidths is None:
        linewidths = np.ones((n_rows, n_cols)) * 1.5
    linewidths = np.asarray(linewidths, dtype=float)  # ensure it's a numpy array for indexing
    # Reorder linewidths from original data order to the visual (clustered) order.
    linewidths_reordered = linewidths[np.ix_(row_order, col_order)]

    box_kws = {
        'edgecolors': edgecolors_reordered,
        'linewidths': linewidths_reordered,
        'background_alpha': background_alpha,
    }

    # Draw the boxes on top of the heatmap mesh
    patch_collection = _overlay_boxes(
        ax=heat_ax,
        heatmap_df=heat_df,
        face_colors=facecolors_reordered,
        sizes=sizes_reordered,
        box_kws=box_kws,
    )

    if legend is not None:
        legend_kws = dict(legend)
        ax_legend = legend_kws.get('ax')
        if ax_legend is None:
            # Narrow column just outside the heatmap's bottom-right corner.
            ax_legend = heat_ax.inset_axes([1.05, 0, 0.15, 0.3])
        ax_legend.set_label('marker_legend')
        _draw_box_legend(ax_legend, heat_ax, sizes_reordered, legend_kws)

    return patch_collection
