# import pandas as pd
# import numpy as np
from matplotlib import pyplot as plt
# import seaborn as sns

def add_pvalue(x_ticks, text, y_left=None, y_top=None, y_right=None, ax=None, line_kw=None, **kwargs):
    """Link two x-ticks and place a text (often the p-value) over the link
    
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
    link.text = ax.text(sum(x_ticks) / 2, y_top * 1.0, text, va='bottom', ha='center', **kwargs)

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




