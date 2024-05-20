# import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
# import seaborn as sns

def link(x_ticks, text, y_left=None, y_top=None, y_right=None, ax=None, line_kw=None, text_kw=None):
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
        ax.boxplot(x=[range(100), range(40, 140)], positions=[0, 1])
        add_pvalue([0, 1], 'test p-value = string', 130, 170, 150, ax=ax)
        plt.show()
    ```
    """
    
    # __version__ = '0.0.1'

    if ax is None:
        ax = plt.gca()

    # setting line defaults
    if line_kw is None:
        line_kw = {
            'linestyle': '-',
            'color': 'k',
            'linewidth': 0.5,
        }
    y_lim = ax.get_ylim()
    y_step = (y_lim[1] - y_lim[0]) / 50
    if y_left is None:
        y_left = y_lim[1] - y_step * 4
    if y_right is None:
        y_right = y_left
    if y_top is None:
        y_top = max(y_left, y_right) + y_step * 2
    
    # draw the lines
    line_x = [x_ticks[0], x_ticks[0], x_ticks[1], x_ticks[1]]
    line_y = [y_left, y_top, y_top, y_right]
    link = ax.plot(line_x, line_y, **line_kw)[0]

    # adding text
    link.text = ax.text(sum(x_ticks) / 2, y_top * 1.0, text, va='bottom', ha='center', **text_kw)

    # adjust y-lim if needed
    if y_top + y_step * 3 > y_lim[1]:
        ax.set_ylim([y_lim[0], y_top + y_step * 3])
    
    return link

    # if type(data) is str:
    #     text = data
    # else:
    #     # * is p < 0.05
    #     # ** is p < 0.005
    #     # *** is p < 0.0005
    #     # etc.
    #     text = ''
    #     p = .05

    #     while data < p:
    #         text += '*'
    #         p /= 10.

    #         if maxasterix and len(text) == maxasterix:
    #             break

    #     if len(text) == 0:
    #         text = 'n. s.'

def forest_plot(
        estimates, 
        origin=0, 
        estimate_labels=None, 
        p_values=None, 
        counts=None,
        line_ys=None,
        line_labels=None, 
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
        counts (_type_, optional): _description_. Defaults to None.
        line_ys (_type_, optional): determines the location of each estimate. Defaults to None.
        line_labels (_type_, optional): _description_. Defaults to None.
        line_colors (_type_, optional): _description_. Defaults to None.
        marker_colors (_type_, optional): _description_. Defaults to None.

    Returns:
        plt.axe: The ax on which the Forest Plot is drawn. Several properties are added:
            * `err_lines`: Is a list of `line` handles.
            * `origin`:
    
    Example:
        ```python
        import pandas as pd

        estimates = pd.DataFrame({
            'estimate': [0.5, 0.6, 0.7, 0.8],
            'conf.low': [0.45, 0.70, 0.90, 0.81],
            'conf.high': [0.55, 0.50, 0.50, 0.79],
        })
        ax = forest_plot(
            estimates=estimates,
            origin=1,
            counts=[10, 20, 30, 40],
            line_ys = np.array([10, 20, 30, 40]),
            line_labels=[f'row{i}' for i in range(4)],
            p_values=np.linspace(0, 0.1, 4),
            estimate_labels=estimates.estimate.map(str),
            # ax=ax,
        )
        ax.set_xlim([0.3, 2])
        print(ax.err_lines)
        plt.show()
        ```
            
    """
    from matplotlib.ticker import ScalarFormatter
    import matplotlib.transforms as transforms

    expected = np.array(estimates['estimate'])
    n_line = len(expected)

    if line_ys is None:
        line_ys = np.arange(n_line)

    # calculate error-bound coordinates
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

    # tranforming axes coordinates to data coordinates:
    # this will do the transform, but if xlim/ylim changes later on, it does not update the coord
    # coord_ax2data = ax.transAxes + ax.transData.inverted()
    # x_right_margin = coord_ax2data.transform((1.02, 0))[0]
    trans_axes_data = transforms.blended_transform_factory(ax.transAxes, ax.transData)

    # ax.invert_yaxis()
    ax.set_ylim([line_ys.max() + 0.5, line_ys.min() - 0.5])

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

    # add estimates labels
    if estimate_labels is not None:
        estimate_labels = np.array(estimate_labels)
        for ri, y_pos in enumerate(line_ys):
            ax.text(expected[ri], y_pos + 0.25, estimate_labels[ri], va='top', ha='center', fontsize=8)

    # add pvalue
    if p_values is not None:
        p_values = np.array(p_values)
        for ri, y_pos in enumerate(line_ys):
           pval_color = 'red' if p_values[ri] <= 0.05 else 'gray'
           ax.text(1.02, y_pos, f'p={p_values[ri]:0.2g}', va='center', ha='left', color=pval_color, fontsize=8, transform=trans_axes_data)

    # add counts
    if counts is not None:
        counts = np.array(counts, dtype=int)
        for ri, y_pos in enumerate(line_ys):
           ax.text(-0.03, y_pos + 0.35, f'n={counts[ri]}', va='center', ha='right', color='#444444', fontsize=8, transform=trans_axes_data)

    # add line labels
    if line_labels is not None:
        ax.set_yticks(line_ys, line_labels)
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(ScalarFormatter())

    return ax

if __name__ == '__main__':
    pass
    # import pandas as pd
    # fig = plt.figure()
    # ax = fig.gca()

    # estimates = pd.DataFrame({
    #     'estimate': [0.5, 0.6, 0.7, 0.8],
    #     'conf.low': [0.45, 0.70, 0.90, 0.81],
    #     'conf.high': [0.55, 0.50, 0.50, 0.79],
    # })
    # ax = forest_plot(
    #     estimates=estimates,
    #     origin=1,
    #     counts=[10, 20, 30, 40],
    #     # line_ys = np.array([1, 0, 3, 2]) + 0.5,
    #     line_labels=[f'row{i}' for i in range(4)],
    #     p_values=np.linspace(0, 0.1, 4),
    #     estimate_labels=estimates.estimate.map(str),
    #     # ax=ax,
    # )
    # ax.set_xlim([0.3, 2])
    # print(ax.err_lines)
    # plt.show()

