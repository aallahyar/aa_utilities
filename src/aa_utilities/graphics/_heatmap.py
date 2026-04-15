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
    
    Args:
        ax: The matplotlib Axes to draw the legend on.
        sizes (np.ndarray): The per-cell sizes matrix (values in [0, 1]).
        legend_kws (dict): Configuration for the legend. Keys:
            - bins (int or array-like): int = number of evenly spaced sizes,
              array-like = explicit size values. Default: 4.
            - labels (list[str]): Explicit label text per bin. Overrides label_fmt.
            - label_fmt (str): Format string for auto-generated labels. Default: '{:.2f}'.
            - title (str): Legend title. Default: None.
            - loc (str): Legend position. Default: 'upper right'.
            - bbox_to_anchor (tuple): Anchor point for placement. Default: None.
            - fontsize (int): Label font size. Default: 8.
            - title_fontsize (int): Title font size. Default: 9.
            - facecolor (str): Fill color for sample boxes. Default: '#cccccc'.
            - edgecolor (str): Border color for sample boxes. Default: '#333333'.
            - frameon (bool): Whether to draw a frame. Default: True.
    """
    from matplotlib import patches
    from matplotlib.legend_handler import HandlerPatch

    # resolve bins
    bins = legend_kws.get('bins', 4)
    if np.ndim(bins) == 0:
        bin_values = np.linspace(sizes.min(), sizes.max(), int(bins))
    else:
        bin_values = np.asarray(bins)

    # resolve labels
    labels = legend_kws.get('labels', None)
    label_fmt = legend_kws.get('label_fmt', '{:.2f}')
    if labels is None:
        labels = [label_fmt.format(v) for v in bin_values]

    # legend styling
    facecolor = legend_kws.get('facecolor', '#cccccc')
    edgecolor = legend_kws.get('edgecolor', '#333333')

    # build proxy artists — one Rectangle per bin, scaled in size
    max_size = bin_values.max() if bin_values.max() > 0 else 1.0
    proxy_artists = []
    for v in bin_values:
        scale = v / max_size
        proxy_artists.append(
            patches.FancyBboxPatch(
                (0, 0), width=scale, height=scale,
                boxstyle='square,pad=0',
                facecolor=facecolor,
                edgecolor=edgecolor,
            )
        )

    # custom handler that draws each proxy at its proportional size
    class _SizedBoxHandler(HandlerPatch):
        def __init__(self, scale, **kwargs):
            self._scale = scale
            super().__init__(**kwargs)

        def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
            side = height * self._scale
            x = xdescent + (width - side) / 2
            y = ydescent + (height - side) / 2
            patch = patches.FancyBboxPatch(
                (x, y), width=side, height=side,
                boxstyle='square,pad=0',
                facecolor=orig_handle.get_facecolor(),
                edgecolor=orig_handle.get_edgecolor(),
                transform=trans,
            )
            return [patch]

    handler_map = {
        artist: _SizedBoxHandler(scale=v / max_size)
        for artist, v in zip(proxy_artists, bin_values)
    }

    legend = ax.legend(
        handles=proxy_artists,
        labels=labels,
        handler_map=handler_map,
        title=legend_kws.get('title', None),
        loc=legend_kws.get('loc', 'upper right'),
        bbox_to_anchor=legend_kws.get('bbox_to_anchor', None),
        fontsize=legend_kws.get('fontsize', 8),
        title_fontproperties={'size': legend_kws.get('title_fontsize', 9)},
        frameon=legend_kws.get('frameon', True),
    )
    ax.add_artist(legend)


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
