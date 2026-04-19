import textwrap

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


def _extract_face_colors(ax, heatmap_df):
    """Extract per-cell RGBA colors from the seaborn heatmap QuadMesh."""
    face_colors = ax.collections[0].get_facecolors()
    if face_colors is None or len(face_colors) == 0:
        face_colors = ax.collections[0]._facecolors
    return face_colors.reshape(*heatmap_df.shape, 4)


def _overlay_boxes(ax, heatmap_df, face_colors, sizes, box_kws):
    """Draw sized rectangles on top of the heatmap mesh and dim the original."""
    from matplotlib import patches
    from matplotlib.collections import PatchCollection

    edgecolors = box_kws.get('edgecolors', np.empty_like(heatmap_df, dtype=object))
    linewidths = box_kws.get('linewidths', np.ones_like(heatmap_df, dtype=float))
    background_alpha = box_kws.get('background_alpha', 0.05)

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
    # Keep a reference to the original QuadMesh before adding the new collection.
    quad_mesh = ax.collections[0]
    ax.add_collection(patch_collection)
    quad_mesh.set_alpha(background_alpha)
    
    return patch_collection


def _draw_box_legend(ax, sizes, legend_kws):
    """Draw a marker-range legend showing representative box sizes.

    The marker axes xlim/ylim are set proportionally to the heatmap axes
    so that 1 data-unit maps to the same physical size on both.  A v×v
    rectangle drawn here is pixel-identical to one on the heatmap.

    Args:
        ax: Axes to draw into (a GridSpec subplot).  Must share a figure
            with an axes labelled ``'heatmap'``.
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
    if np.ndim(bins) == 0:
        bin_values = np.linspace(sizes.min(), sizes.max(), int(bins))
    else:
        bin_values = np.asarray(bins)
    n_bins = len(bin_values)

    labels = legend_kws.get('labels', None)
    label_fmt = legend_kws.get('label_fmt', '{:.2f}')
    if labels is None:
        labels = [label_fmt.format(v) for v in bin_values]

    facecolor = legend_kws.get('facecolor', '#cccccc')
    edgecolor = legend_kws.get('edgecolor', '#333333')
    fontsize = legend_kws.get('fontsize', 8)
    title_fontsize = legend_kws.get('title_fontsize', 9)
    title = legend_kws.get('title', None)
    linewidth = legend_kws.get('linewidth', 1)

    # Match data-to-pixel scale with the heatmap so v×v boxes look identical.
    ax_heat = next(a for a in ax.get_figure().axes if a.get_label() == 'heatmap')
    gs = ax.get_subplotspec().get_gridspec()
    w_ratios, h_ratios = gs.get_width_ratios(), gs.get_height_ratios()

    heat_xrange = np.abs(np.diff(ax_heat.get_xlim()))[0]
    heat_yrange = np.abs(np.diff(ax_heat.get_ylim()))[0]
    marker_xmax = heat_xrange * w_ratios[1] / w_ratios[0]
    marker_ymax = heat_yrange * h_ratios[1] / sum(h_ratios)

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
        cy = mi * spacing + spacing / 2 + spacing / 10   # center y of the current marker
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
            cy, textwrap.fill(lbl, width=10),
            va='center_baseline', ha='left', fontsize=fontsize,
        )

    if title:
        ax.set_title(
            title,
            va='center', ha='center', fontsize=title_fontsize,
            fontweight='normal',
        )

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.patch.set_visible(False)


def heatmap(matrix_df, box_kws, fig=None, gs_kws=None, **kwargs):
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
            - linewidths (np.ndarray): Per-cell border widths. Default: 1.5.
            - background_alpha (float): Original heatmap (mesh) alpha. Default: 0.05.
            - legend (dict): Marker legend config passed to
              ``_draw_box_legend``.  Default: ``{}`` (draws a 4-bin auto
              legend).
        fig (matplotlib.figure.Figure, optional): Pre-created figure to draw
            into.  If ``None``, a new ``(10, 8)`` figure is created.
        gs_kws (dict): Keyword arguments forwarded to
            ``fig.add_gridspec()``.  Use this to control the layout, e.g.
            ``{'width_ratios': [10, 0.5], 'height_ratios': [1, 1],
            'wspace': 0.3, 'hspace': 0.5}``.
        **kwargs: Forwarded to ``sns.heatmap()``.  The ``ax`` key is ignored;
            ``cbar_ax`` is overridden to point at the colorbar subplot.

    Returns:
        matplotlib.figure.Figure

    Example::

        import numpy as np, pandas as pd
        from matplotlib import colors

        corr = pd.DataFrame(np.arange(-112, 113).reshape(15, 15) / 224)
        cmap = colors.LinearSegmentedColormap.from_list(
            'BuWtRd', ['blue', 'white', 'red'], N=8)
        fig = heatmap(
            corr,
            box_kws={
                'sizes': corr.abs().values * 0.98 / corr.values.max(),
                'linewidths': np.ones_like(corr, dtype=float) * 0.9,
                'background_alpha': 0.7,
                'edgecolors': np.where(corr.le(0), '#000000', None),
                'legend': {'bins': 4, 'title': 'Box size'},
            },
            cmap=cmap,
            cbar_kws={'label': 'Correlation'},
        )
        fig.show()
    """

    gs_kws = dict(gs_kws or {})
    gs_kws.setdefault('width_ratios', [10, 0.5])
    gs_kws.setdefault('height_ratios', [1, 1])
    gs_kws.setdefault('hspace', 0.5)
    gs_kws.setdefault('wspace', 0.1)

    if fig is None:
        fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 2, **gs_kws)

    ax_heat = fig.add_subplot(gs[:, 0])
    ax_cbar = fig.add_subplot(gs[0, 1])
    ax_marker = fig.add_subplot(gs[1, 1])

    ax_heat.set_label('heatmap')
    ax_cbar.set_label('colorbar')
    ax_marker.set_label('marker_legend')

    # Redirect colorbar to its dedicated axes; ignore user-supplied ax
    kwargs.pop('ax', None)
    kwargs['cbar_ax'] = ax_cbar

    sns.heatmap(data=matrix_df, ax=ax_heat, **kwargs)

    # Overlay boxes
    face_colors = _extract_face_colors(ax_heat, matrix_df)
    sizes = box_kws.get('sizes', np.ones(matrix_df.shape) * 0.8)
    _overlay_boxes(ax_heat, matrix_df, face_colors, sizes, box_kws)

    # Draw marker legend
    legend_kws = box_kws.get('legend', {})
    if legend_kws is None:
        legend_kws = {}
    _draw_box_legend(ax_marker, sizes, legend_kws)

    return fig


def overlay_boxes(
        clustermap_obj,
        sizes=None,
        edgecolors=None,
        linewidths=None,
        background_alpha=0.1,
    ):
    """Overlay sized rectangles on an existing seaborn clustermap's heatmap.

    Args:
        clustermap_obj: The ``ClusterGrid`` object returned by
            ``sns.clustermap()``.
        sizes (np.ndarray, optional): Per-cell box sizes in [0, 1], in
            **original** data order (pre-clustering). 
            Default: 0.8 uniform.
        edgecolors (np.ndarray, optional): Per-cell edge colors in any valid
            matplotlib format (e.g., '#RRGGBB', (r, g, b), etc.), in **original** 
            data order. 
            Default: no (i.e., fully transparent) edge color.
        linewidths (np.ndarray, optional): Per-cell border widths in **original** data order. Default: 1.5.
        background_alpha (float): Heatmap mesh alpha after dimming.
            Default: 0.1.

    Returns:
        matplotlib.collections.PatchCollection: The overlay patch collection.

    Notes:
        When ``row_cluster=False`` or ``col_cluster=False`` was passed to
        ``sns.clustermap()``, the corresponding dendrogram is ``None`` and
        the original row/column order is used as-is.
    """
    ax_heat = clustermap_obj.ax_heatmap

    # data2d has already been reordered to the clustered (visual) layout by seaborn's
    # plot_matrix() before clustermap() returns, so it matches the heatmap's rendered order.
    heatmap_df = clustermap_obj.data2d

    n_rows, n_cols = heatmap_df.shape

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

    if sizes is None:
        sizes = np.ones((n_rows, n_cols)) * 0.8
    # Reorder sizes from original data order to the visual (clustered) order.
    sizes_reordered = sizes[np.ix_(row_order, col_order)]

    if edgecolors is None:
        # default to no edge color
        edgecolors = np.empty((n_rows, n_cols), dtype=object)
    # Reorder edgecolors from original data order to the visual (clustered) order.
    edgecolors_reordered = edgecolors[np.ix_(row_order, col_order)]

    if linewidths is None:
        linewidths = np.ones((n_rows, n_cols)) * 1.5
    # Reorder linewidths from original data order to the visual (clustered) order.
    linewidths_reordered = linewidths[np.ix_(row_order, col_order)]

    box_kws = {
        'edgecolors': edgecolors_reordered,
        'linewidths': linewidths_reordered,
        'background_alpha': background_alpha,
    }

    face_colors = _extract_face_colors(ax_heat, heatmap_df)
    patch_collection = _overlay_boxes(ax_heat, heatmap_df, face_colors, sizes_reordered, box_kws)

    return patch_collection