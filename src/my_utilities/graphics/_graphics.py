# import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
# import seaborn as sns

def link(x_ticks, text, y_left, y_top=None, y_right=None, ax=None, line_kw=None, text_kw=None):
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

    Returns:
    -------
    ax: matplotlib ax
        ax with the added annotation
    '''

    Example:
    -------
    ```python
        fig = plt.figure()
        ax = fig.gca()
        ax.boxplot(x=[np.linspace(1, 100), np.linspace(40, 140)], positions=[0, 1])
        # ax.set_yscale('log', base=10)
        ax.set_ylim([1, 100])
        link([0, 1], text='test p-value = string', y_left=100, y_right=140, ax=ax)
        plt.show()
    ```
    """
    
    # __version__ = '0.0.1'
    import matplotlib.transforms as mpl_transforms

    if ax is None:
        ax = plt.gca()

    # setup coordinate offsets
    y_offset = mpl_transforms.offset_copy(ax.transData, y=+12, units='dots')
    dsp2dt = ax.transData.inverted()

    # setting defaults
    if line_kw is None:
        line_kw = {
            'linestyle': '-',
            'color': 'k',
            'linewidth': 0.5,
        }
    
    # setting text defaults
    if text_kw is None:
        text_kw = {
        }
    
    # setting y defaults
    if y_right is None:
        y_right = y_left
    if y_top is None:
        y_top = max(y_left, y_right)
    
    # bring top coordinate one step up
    y_top_adj = dsp2dt.transform(y_offset.transform((1, y_top)))[1]
    
    # draw the lines
    line_x = [x_ticks[0], x_ticks[0], x_ticks[1], x_ticks[1]]
    line_y = [y_left, y_top_adj, y_top_adj, y_right]
    link = ax.plot(line_x, line_y, transform=y_offset, **line_kw)[0]

    # adding text
    link.text = ax.text(sum(x_ticks) / 2, y_top_adj, text, transform=y_offset, va='bottom', ha='center', **text_kw)

    # adjust y-lim if needed
    y_max = y_top_adj + (y_top_adj - y_top) * 5
    y_lim = list(ax.get_ylim())
    if y_max > y_lim[1]:
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
            ax=ax,
        )
        ax.set_xlim([0.3, 2])
        plt.show()
        ```
            
    """
    from matplotlib.ticker import ScalarFormatter
    import matplotlib.transforms as mpl_transforms

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
    count_offset = mpl_transforms.offset_copy(trans_axes_data, x=-10, y=-20, units='dots')
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
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(ScalarFormatter())

    # add line sub-labels
    if line_sublabels is not None:
        line_sublabels = np.array(line_sublabels)
        for ri, y_pos in enumerate(line_ys):
           ax.err_lines[ri].sublabel = ax.text(0, y_pos, line_sublabels[ri], va='center', ha='right', color='#444444', fontsize=8, transform=count_offset)

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
           ax.text(1, y_pos, f'p={p_values[ri]:0.2g}', va='center', ha='left', color=pval_color, fontsize=8, transform=pval_offset)

    return ax

if __name__ == '__main__':
    pass

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
        ax=ax,
    )
    ax.set_xlim([0.3, 2])
    plt.show()

