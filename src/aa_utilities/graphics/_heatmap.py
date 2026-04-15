import numpy as np
from matplotlib import pyplot as plt


def _extract_face_colors(ax, matrix_df):
    """Extract per-cell RGBA colors from the seaborn heatmap QuadMesh."""
    # infer properties
    # cmap = ax.collections[0].cmap
    # vmin = ax.collections[0].colorbar.vmin
    # vmax = ax.collections[0].colorbar.vmax
    # color_idxs = (matrix_df.values - vmin) / (vmax - vmin)
    # face_colors = cmap(color_idxs)
    face_colors = ax.collections[0].get_facecolors() # shape: (n_row * n_col, 4)
    if face_colors is None or len(face_colors) == 0:
        # Fallback for older matplotlib where facecolors may be empty until draw
        # shape of _facecolors is (n_row, n_col, 4)
        face_colors = ax.collections[0]._facecolors
    return face_colors.reshape(*matrix_df.shape, 4) # unifying shape


def _overlay_boxes(ax, matrix_df, face_colors, sizes, box_kws):
    """Draw sized rectangles on top of the heatmap mesh and dim the original."""
    from matplotlib import patches
    from matplotlib.collections import PatchCollection

    edgecolors = box_kws.get('edgecolors', np.empty_like(matrix_df, dtype=object))
    linewidth = box_kws.get('linewidth', 3)
    background_alpha = box_kws.get('background_alpha', 0.05)

    rectangles = []
    for ri, row in enumerate(matrix_df.index):
        for ci, col in enumerate(matrix_df.columns):
            rectangles.append(
                patches.Rectangle(
                    (ci + 0.5 - sizes[ri, ci] / 2, ri + 0.5 - sizes[ri, ci] / 2),
                    width=sizes[ri, ci],
                    height=sizes[ri, ci],
                    facecolor=face_colors[ri, ci],
                    edgecolor=edgecolors[ri, ci],
                    linewidth=linewidth,
                )
            )
    ax.add_collection(PatchCollection(rectangles, match_original=True))

    # adjust the original heatmap's alpha allows more visible boxes (if background_alpha < 1)
    ax.collections[0].set_alpha(background_alpha)
    # ax.xaxis.tick_top()


def _draw_box_legend(ax, sizes, legend_kws):
    """Draw a legend showing representative box sizes on a heatmap axes.
    
    Draws rectangles using the exact same sizing mechanism as _overlay_boxes
    (patches.Rectangle in data coordinates) inside a floating axes, so the
    legend boxes are pixel-identical to the heatmap boxes.
    
    Args:
        ax: The matplotlib Axes to draw the legend on.
        sizes (np.ndarray): The per-cell sizes matrix (values in [0, 1]).
        legend_kws (dict): Configuration for the legend. Keys:
            - bins (int or array-like): int = number of evenly spaced sizes,
              array-like = explicit size values. Default: 4.
            - labels (list[str]): Explicit label text per bin. Overrides label_fmt.
            - label_fmt (str): Format string for auto-generated labels. Default: '{:.2f}'.
            - title (str): Legend title. Default: None.
            - position (tuple): (x, y) in axes fraction for the legend's lower-left corner.
              Default: (1.02, 0.5) — just outside the right edge, vertically centered.
            - fontsize (int): Label font size. Default: 8.
            - title_fontsize (int): Title font size. Default: 9.
            - facecolor (str): Fill color for sample boxes. Default: '#cccccc'.
            - edgecolor (str): Border color for sample boxes. Default: '#333333'.
            - frameon (bool): Whether to draw a frame. Default: True.
    """
    from matplotlib import patches

    # resolve bins
    bins = legend_kws.get('bins', 4)
    if np.ndim(bins) == 0:
        bin_values = np.linspace(sizes.min(), sizes.max(), int(bins))
    else:
        bin_values = np.asarray(bins)
    n_bins = len(bin_values)

    # resolve labels
    labels = legend_kws.get('labels', None)
    label_fmt = legend_kws.get('label_fmt', '{:.2f}')
    if labels is None:
        labels = [label_fmt.format(v) for v in bin_values]

    # styling
    facecolor = legend_kws.get('facecolor', '#cccccc')
    edgecolor = legend_kws.get('edgecolor', '#333333')
    fontsize = legend_kws.get('fontsize', 8)
    title_fontsize = legend_kws.get('title_fontsize', 9)
    title = legend_kws.get('title', None)
    frameon = legend_kws.get('frameon', True)

    # Compute heatmap cell size in figure-fraction units so the legend axes
    # uses the same scale as the heatmap — 1 data unit in the legend axes
    # equals 1 data unit in the heatmap axes.
    fig = ax.get_figure()
    fig_w, fig_h = fig.get_size_inches()
    fig_w_px, fig_h_px = fig_w * fig.dpi, fig_h * fig.dpi

    p0 = ax.transData.transform((0, 0))
    p1 = ax.transData.transform((1, 1))
    cell_w_px = abs(p1[0] - p0[0])
    cell_h_px = abs(p1[1] - p0[1])

    # Convert cell size to figure fractions
    cell_w_frac = cell_w_px / fig_w_px
    cell_h_frac = cell_h_px / fig_h_px

    # Estimate label width in figure fractions
    max_label_len = max(len(lbl) for lbl in labels)
    label_w_px = fontsize * max_label_len * 0.6 * fig.dpi / 72.0
    padding_px = fontsize * 0.5 * fig.dpi / 72.0
    label_cols = (label_w_px + padding_px) / cell_w_px  # in cell-width units

    # Legend grid: 1 column for boxes + label_cols for text, n_bins rows + optional title
    title_rows = 1 if title else 0
    n_rows = n_bins + title_rows
    legend_w_frac = cell_w_frac * (1 + label_cols)
    legend_h_frac = cell_h_frac * n_rows

    # Position: default just outside the right edge of the heatmap axes
    position = legend_kws.get('position', (1.02, 0.5))
    ax_bbox = ax.get_position()
    legend_x = ax_bbox.x0 + position[0] * ax_bbox.width
    legend_y = ax_bbox.y0 + position[1] * ax_bbox.height - legend_h_frac / 2

    # Create a floating axes with the same cell scale as the heatmap
    legend_ax = fig.add_axes([legend_x, legend_y, legend_w_frac, legend_h_frac])
    legend_ax.set_xlim(0, 1 + label_cols)
    legend_ax.set_ylim(0, n_rows)
    legend_ax.set_aspect('equal')
    legend_ax.axis('off')

    # Draw rectangles using the same formula as _overlay_boxes
    for i, (v, lbl) in enumerate(zip(bin_values, labels)):
        cx, cy = 0.5, i + 0.5  # cell center (row 0 = bottom)
        legend_ax.add_patch(
            patches.Rectangle(
                (cx - v / 2, cy - v / 2),
                width=v,
                height=v,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=legend_kws.get('linewidth', 1),
            )
        )
        legend_ax.text(
            1 + padding_px / cell_w_px * 0.5, cy, lbl,
            va='center', ha='left', fontsize=fontsize,
        )

    # Add title
    if title:
        legend_ax.text(
            (1 + label_cols) / 2, n_rows - 0.5, title,
            va='center', ha='center', fontsize=title_fontsize,
            fontweight='bold',
        )

    # Frame
    if frameon:
        for spine in legend_ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('#cccccc')
            spine.set_linewidth(0.5)
    
    return legend_ax


def heatmap(matrix_df, box_kws=None, **kwargs):
    """Plots a heatmap with optional sized/bordered rectangles per cell.

    Note: Colorbar handle is located at: ax.collections[0].colorbar.cmap

    Args:
        matrix_df (pd.DataFrame): Matrix for which a heatmap will be drawn.
        box_kws (dict, optional): If provided, overlay rectangles on each cell. Keys:
            - sizes (np.ndarray): Per-cell box sizes in [0, 1]. Default: 0.8 uniform.
            - edgecolors (np.ndarray): Per-cell edge colors. Default: None.
            - linewidth (float): Rectangle border width. Default: 3.
            - background_alpha (float): Alpha of the original heatmap mesh. Default: 0.05.
            - legend (dict or None): If provided, draw a size legend. See _draw_box_legend.
        **kwargs: Passed directly to sns.heatmap().
    
    Returns:
        ax: The matplotlib Axes with the heatmap drawn.

    Example:
            import numpy as np
            import pandas as pd
            from matplotlib import pyplot as plt, colors

            # data preparation
            # rng = np.random.default_rng(seed=42)
            # corr_df = pd.DataFrame(rng.uniform(-1, 1, size=(15, 15)))
            corr_df = pd.DataFrame(np.arange(-112, 113).reshape(15, 15) / 224)
            corr_df.index = corr_df.index.map(lambda i: 'row{:d}'.format(i))
            corr_df.columns = corr_df.columns.map(lambda c: 'col{:d}'.format(c))

            fig = plt.figure(figsize=(7, 6))
            ax = fig.gca()
            cmap = colors.LinearSegmentedColormap.from_list('BlueWhiteRed', ['blue', 'white', 'red'], N=8, gamma=1.0)
            heatmap(
                corr_df, 
                cmap=cmap, 
                ax=ax,
                cbar_kws={
                    'label': 'Label of the colorbar', 
                    'extend': 'max',
                    'fraction': 0.15,   # increase to allocate more space to the colorbar axis
                    'shrink': 0.5,      # closer to 1.0 means less shrinking (thicker bar)
                    'aspect': 10,       # smaller aspect makes the bar thicker (for vertical bars)
                    'ticks': [-4, 0, 4],
                },
                box_kws={
                    'background_alpha': 0.7, # background alpha of the each element in the heatmap (same as `facecolor`)
                    'sizes': corr_df.abs().values * 0.98 / corr_df.values.max(), 
                    'linewidth': 0.9, 
                    'edgecolors': np.where(corr_df.le(0), '#000000', None),
                    'legend': {
                        'bins': 4,
                        'title': 'Box size',
                    },
                },
            )

            # sanity check:
            # sns.heatmap(corr_df, cmap=cmap, ax=ax)

            plt.show()
    """
    import seaborn as sns
    # cmap = sns.color_palette('vlag', n_colors=4, as_cmap=True)
    # cmap = colors.LinearSegmentedColormap.from_list('BlueWhiteRed', ['blue', 'white', 'red'], N=20, gamma=1.0)
    
    ax = sns.heatmap(
        data=matrix_df,
        # mask=corr_df.abs() < 0.5, # hide elements
        # annot=True,
        # annot_kws={'fontsize': 6},
        # fmt='.1f',
        # cmap=cmap,
        # center=0,
        # vmin=-1,
        # vmax=+1,
        # cbar_kws={'label': 'Label of the colorbar', 'extend': 'max', 'fraction': 0.15, 'shrink': 0.5, 'aspect': 10, 'ticks': [-4, 0, 4]},
        **kwargs,
    )

    if box_kws is not None:
        face_colors = _extract_face_colors(ax, matrix_df)
        sizes = box_kws.get('sizes', np.ones_like(matrix_df) * 0.8)
        _overlay_boxes(ax, matrix_df, face_colors, sizes, box_kws)

        legend_kws = box_kws.get('legend')
        if legend_kws is not None:
            _draw_box_legend(ax, sizes, legend_kws)

    return ax
