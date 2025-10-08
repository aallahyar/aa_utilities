# import pandas as pd
import numpy as np
from matplotlib import (
    pyplot as plt,
    transforms as mpl_transforms,
)
# import seaborn as sns


#%%
def link(x_ticks, text, y_left, y_top=None, y_right=None, height=10, pad=5, top_space=10, ax=None, line_kws=None, text_kws=None):
    """Links two x-ticks and places a text (e.g., a p-value for a test) over the link
    
    Parameters:
    ----------
    ax: matplotlib Axes
        handle of the axis on which the annotation is going to be placed on
    text: str
        string that will be placed over the link line
    x_ticks: list
        a 2-element list of x-tick indices that needs to be annotated.
        e.g.: [1,2] links column index 1 to 2
    pad: int
        Space between given y_left and link's lines
    top_space: int
        Space between link's top and the top of the axes

    Returns:
    -------
    ax: matplotlib ax
        ax with the added annotation
    

    Example:
    -------
    ```python
        fig = plt.figure()
        ax = fig.gca()
        ax.boxplot(x=[np.linspace(1, 100), np.linspace(40, 140)], positions=[0, 1])
        ax.set_yscale('log', base=10)
        # ax.set_ylim(top=1e10)

        y = 100
        for i in range(10):
            link_h = link([0, 1], text='test p-value = string', y_left=y, y_right=y + 40, ax=ax)

            coords_disp = link_h.text.get_window_extent()
            coords_disp.y1 += 0 # offset +10 points to the last p-value text, in display coordinates
            coords_data = ax.transData.inverted().transform(coords_disp) # transform from display to data coordinates
            # ax.plot([0, 4], coords_data[1, [1, 1]])
            y = coords_data[1, 1]
        plt.show()
    ```
    """
    
    if ax is None:
        ax = plt.gca()

    # setup coordinate offsets
    transform_pad = mpl_transforms.offset_copy(ax.transData, y=pad, units='dots')
    transform_height = mpl_transforms.offset_copy(ax.transData, y=height, units='dots')

    # setting defaults
    if line_kws is None:
        line_kws = {
            'linestyle': '-',
            'color': '#555555',
            'linewidth': 0.5,
        }
    
    # setting text defaults
    if text_kws is None:
        text_kws = {
            'color': '#555555',
        }
    
    # setting y defaults
    if y_right is None:
        y_right = y_left
    if y_top is None:
        y_max = max(y_left, y_right)
        y_top = ax.transData.inverted().transform(transform_height.transform((1, y_max)))[1]
    
    # draw the lines
    line_x = [x_ticks[0], x_ticks[0], x_ticks[1], x_ticks[1]]
    line_y = [y_left, y_top, y_top, y_right]
    link = ax.plot(line_x, line_y, transform=transform_pad, **line_kws)[0]

    # adding text
    link.text = ax.text(sum(x_ticks) / 2, y_top, text, transform=transform_pad, va='bottom', ha='center', **text_kws)

    # adjust y-lim if needed
    if top_space is not None:
        coords_disp = link.text.get_window_extent()
        coords_disp.y1 += top_space # go higher from the text, in display coordinates
        coords_data = ax.transData.inverted().transform(coords_disp) # transform from display to data coordinates
        y_max = coords_data[1, 1]
        if y_max > ax.get_ylim()[1]:
            ax.set_ylim(top=y_max)
    
    return link

def forest_plot(
        estimates, 
        origin=0, 
        estimate_labels=None, 
        p_values=None, 
        line_ys=None,
        line_labels=None, 
        line_sublabels=None, 
        line_colors=None, 
        marker_colors=None,
        x_scale=None,
        ax=None,
    ):
    """Generates a ForestPlot that is often used in Treatment-effect presentations

    Args:
        estimates (pd.DataFrame): A dataframe containing the `estimate` column
        origin (int, optional): _description_. Defaults to 0.
        estimate_labels (_type_, optional): _description_. Defaults to None.
        p_values (_type_, optional): _description_. Defaults to None.
        line_ys (_type_, optional): determines the location of each estimate. Defaults to None.
        line_labels (_type_, optional): _description_. Defaults to None.
        line_sublabels (_type_, optional): _description_. Defaults to None.
        line_colors (_type_, optional): _description_. Defaults to None.
        marker_colors (_type_, optional): _description_. Defaults to None.

    Returns:
        plt.axe: The ax on which the Forest Plot is drawn. Several properties are added:
            * `err_lines`: Is a list of `line` handles.
            * `origin`:
    
    Example:
        ```python
        import pandas as pd
        fig = plt.figure(figsize=(2, 3 / 1.7))
        ax = fig.gca()

        estimates = pd.DataFrame({
            'estimate': [0.5, 0.60, 0.7, 0.8],
            'conf.low': [0.45, 0.50, 0.65, 0.79],
            'conf.high': [0.55, 0.70, 0.80, 0.91],
        })
        ax = forest_plot(
            estimates=estimates,
            origin=1,
            line_ys = np.array([1, 0, 3, 2]) + 0.5,
            line_labels=[f'row{i}' for i in range(4)],
            line_sublabels=[f'n={n}' for n in [0, 1, 2, 3]],
            p_values=np.linspace(0, 0.1, 4),
            estimate_labels=estimates.estimate.map(str),
            x_scale=dict(value='log', base=2),
            ax=ax,
        )
        ax.set_xlim([0.3, 2])
        plt.show()
        ```    
    """
    from matplotlib.ticker import ScalarFormatter

    expected = np.array(estimates['estimate'])
    n_line = len(expected)

    if line_ys is None:
        line_ys = np.arange(n_line)
    line_ys = np.array(line_ys)

    # calculate error-bound coordinates
    assert estimates['estimate'].between(estimates['conf.low'], estimates['conf.high'], inclusive='both').all()
    err_ends = np.abs(estimates[['conf.low', 'conf.high']].values - expected[:, None]).T # each column refers to a single line

    if line_colors is None:
        line_colors = np.repeat('#000000', n_line)
    if marker_colors is None:
        marker_colors = line_colors
    line_colors = np.array(line_colors)
    marker_colors = np.array(marker_colors)

    if ax is None:
        fig = plt.figure(figsize=(2, n_line / 1.7))
        ax = fig.gca()

    # generate tranformers
    trans_axes_data = mpl_transforms.blended_transform_factory(ax.transAxes, ax.transData)
    sublabel_offset = mpl_transforms.offset_copy(trans_axes_data, x=-10, y=-15, units='dots')
    estimate_label_offset = mpl_transforms.offset_copy(ax.transData, y=-13, units='dots')
    pval_offset = mpl_transforms.offset_copy(trans_axes_data, x=5, units='dots')

    # ax.invert_yaxis()
    ax.set_ylim(bottom=line_ys.max() + 0.75, top=line_ys.min() - 0.5)

    # add origin
    ax.origin = ax.axvline(x=origin, linestyle=':', color='#cccccc')

    # add error lines
    ax.err_lines = [None] * n_line
    for ri, y_pos in enumerate(line_ys):
        ax.err_lines[ri] = ax.errorbar(
            x=expected[ri], 
            y=y_pos, 
            xerr=err_ends[:, [ri]], 
            color=line_colors[ri],
            elinewidth=1,
            marker='d', 
            markerfacecolor=marker_colors[ri],
            markeredgecolor='none', 
            markersize=10, 
            capsize=3, # error end-cap height
            markeredgewidth=2, # error end-cap width
        )
    
    # add line labels
    if line_labels is not None:
        ax.set_yticks(line_ys, line_labels)
    if x_scale is not None:
        ax.set_xscale(**x_scale)
    ax.xaxis.set_major_formatter(ScalarFormatter())

    # add line sub-labels
    if line_sublabels is not None:
        line_sublabels = np.array(line_sublabels)
        for ri, y_pos in enumerate(line_ys):
           ax.err_lines[ri].sublabel = ax.text(
               0, y_pos, line_sublabels[ri], va='center', ha='right', color='#444444', fontsize=8, transform=sublabel_offset
            )

    # add estimates labels
    if estimate_labels is not None:
        estimate_labels = np.array(estimate_labels)
        for ri, y_pos in enumerate(line_ys):
            ax.text(expected[ri], y_pos, estimate_labels[ri], va='top', ha='center', fontsize=8, transform=estimate_label_offset)

    # add pvalue
    if p_values is not None:
        p_values = np.array(p_values)
        for ri, y_pos in enumerate(line_ys):
           pval_color = 'red' if p_values[ri] <= 0.05 else 'gray'
           ax.text(1, y_pos, f'p={p_values[ri]:0.1g}', va='center', ha='left', color=pval_color, fontsize=8, transform=pval_offset)

    return ax


def add_counts_to_legend(ax, counts, text_format='{label:s} (n={count:d})', **kwargs):
    handles, labels = ax.get_legend_handles_labels()
    n_elements = len(labels)

    for ei in range(n_elements):
        if labels[ei] in counts:
            labels[ei] = text_format.format(label=labels[ei], count=counts[labels[ei]])
    ax.legend(handles, labels, **kwargs)
    return (handles, labels)


def heatmap(matrix_df, **kwargs):
    """Plots a heatmap, but also allows:
        - Customized rectangle sizes per element
        - Line borders per element
        Note: Colorbar handle is located at: ax.collections[0].colorbar.cmap

    Args:
        matrix_df (pd.DataFrame): Matrix for which a heatmap will be drawn
    
    Example:
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
                cbar_kws={'label': 'Label of the colorbar', 'extend': 'max'},
                box_kws={
                    'mesh_alpha': 0.7, # determines the background color of the heatmap (which is the same as facecolor)
                    'sizes': corr_df.abs().values * 0.98 / corr_df.values.max(), 
                    'line_width': 0.9, 
                    'edge_colors': np.where(corr_df.le(0), '#000000', None)
                },
            )

            # sanity check:
            # sns.heatmap(corr_df, cmap=cmap, ax=ax)

            plt.show()
    """
    from matplotlib import (
        # pyplot as plt, 
        patches,
        # colors,
    )
    from matplotlib.collections import PatchCollection
    import seaborn as sns
    
    # initialize default parameters
    box_kws = kwargs.pop('box_kws', {})
    # cmap = sns.color_palette('vlag', n_colors=4, as_cmap=True)
    # cmap = colors.LinearSegmentedColormap.from_list('BlueWhiteRed', ['blue', 'white', 'red'], N=20, gamma=1.0)
    
    ax = sns.heatmap(
        data=matrix_df,
        # mask=corr_df.abs() < 0.5, # hide elements
        # annot=True,
        # annot_kws={'fontsize': 6},
        # fmt='.1f',
        # cmap=cmap,
        # vmin=-1,
        # vmax=+1,
        **kwargs,
    )

    if box_kws is not None:
        
        # infer properties
        # cmap = ax.collections[0].cmap
        # vmin = ax.collections[0].colorbar.vmin
        # vmax = ax.collections[0].colorbar.vmax
        # color_idxs = (matrix_df.values - vmin) / (vmax - vmin)
        # face_colors = cmap(color_idxs)
        face_colors = ax.collections[0]._facecolors
        edge_colors = box_kws.get('edge_colors', np.empty_like(matrix_df, dtype=object))
        sizes = box_kws.get('sizes', np.ones_like(matrix_df) * 0.8)
        line_width = box_kws.get('line_width', 3)
        mesh_alpha = box_kws.get('mesh_alpha', 0.05)
    
        # draw boxes
        rectangles = []
        for ri, row in enumerate(matrix_df.index):
            for ci, col in enumerate(matrix_df.columns):
                rectangles.append(
                    patches.Rectangle(
                        (ci + 0.5 - sizes[ri, ci] / 2, ri + 0.5 - sizes[ri, ci] / 2),
                        width=sizes[ri, ci],
                        height=sizes[ri, ci],
                        facecolor=face_colors[ri, ci],
                        edgecolor=edge_colors[ri, ci],
                        linewidth=line_width,
                    )
                )
        ax.add_collection(PatchCollection(rectangles, match_original=True))

        # final adjustments
        ax.collections[0].set_alpha(mesh_alpha)
        # ax.xaxis.tick_top()
    
    return ax


def text_offset(x, y, text, offsets=(0, -0.02), types=('data', 'ax'), units='points', ax=None, **kwargs):
    """Adds a text using a give offset
    
    Example:
    df4plt = (
        pd.DataFrame()
        .assign(
            x=np.arange(11),
            y=np.linspace(0, 1, num=11),
            color=lambda df: np.where(df.x < 5, 'blue', 'green'),
        )
    )

    fig = plt.figure(figsize=(6, 6))
    ax = fig.gca()
    for i, row in df4plt.iterrows():
        text_offset(row.x, row.y, f'{row.x:3d}, {row.y:3.1f}', color=row.color, offsets=(0, -0.05), ax=ax)

    ax.set_xlim([0, 10])
    ax.set_ylim([0, 100])
    ax.grid()
    plt.show()
    """
    if ax is None:
        ax = plt.gca()

    # define transform
    transform_factory = {'data': ax.transData, 'ax': ax.transAxes}
    transformer = mpl_transforms.blended_transform_factory(transform_factory[types[0]], transform_factory[types[1]])
    trans_offset = mpl_transforms.offset_copy(transformer, x=offsets[0], y=offsets[1], units=units, fig=ax.get_figure())

    # set default text params
    kwargs = dict(va='top', ha='center', transform=trans_offset) | kwargs # fontsize=8, color='black', 

    # draw the text
    text_hndl = ax.text(x, y, text, **kwargs)
    return text_hndl

#%%
if __name__ == '__main__':
    pass

    fig = plt.figure()
    ax = fig.gca()
    ax.boxplot(x=[np.linspace(1, 100), np.linspace(40, 140)], positions=[0, 1])
    ax.set_yscale('log', base=10)
    # ax.set_ylim(top=1e10)

    y = 100
    for i in range(10):
        link_h = link([0, 1], text='test p-value = string', y_left=y, y_right=y + 40, ax=ax)

        coords_disp = link_h.text.get_window_extent()
        coords_disp.y1 += 0 # offset +10 points to the last p-value text, in display coordinates
        coords_data = ax.transData.inverted().transform(coords_disp) # transform from display to data coordinates
        # ax.plot([0, 4], coords_data[1, [1, 1]])
        y = coords_data[1, 1]
    plt.show()

