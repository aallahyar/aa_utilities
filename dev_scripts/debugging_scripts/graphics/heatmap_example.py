
#%% preparation
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, colors
import seaborn as sns

from aa_utilities.graphics import heatmap as aa_heatmap, overlay_boxes

# data preparation
corr_df = pd.DataFrame(np.arange(-112, 113).reshape(15, 15) / 224)
corr_df.index = corr_df.index.map(lambda i: 'row{:d}'.format(i))
corr_df.columns = corr_df.columns.map(lambda c: 'col{:d}'.format(c))
cmap = colors.LinearSegmentedColormap.from_list(
    'BlueWhiteRed', ['blue', 'white', 'red'], N=8, gamma=1.0)
sizes = corr_df.abs().values * 0.98 / corr_df.abs().values.max()
linewidths = np.ones_like(corr_df, dtype=float) * 0.9

#%% sanity check: For reference, plain sns.heatmap
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(corr_df, cmap=cmap, ax=ax)
ax.set_title('Sanity check: For reference, plain sns.heatmap')
fig.show()

#%% example of using overlay_boxes with sns.clustermap
clsmap = sns.clustermap(
    corr_df, 
    row_cluster=False,
    cmap=cmap, 
    figsize=(8, 8),
)
clsmap.figure.canvas.manager.set_window_title('Clustermap, reference')

#%% basic heatmap with boxes + legend, only showing boxes for negative values
fig = aa_heatmap(
    corr_df,
    box_kws={
        'background_alpha': 0.7,
        'sizes': sizes,
        'linewidths': linewidths,
        'edgecolors': np.where(corr_df.le(0), '#000000', 'none'),
        'legend': {
            'bins': 4,
            'title': 'Box size',
        },
    },
    fig=plt.figure(figsize=(10, 8)),
    cmap=cmap,
    cbar_kws={
        'label': 'Label of the colorbar',
        'extend': 'max',
        'ticks': [-0.5, 0, 0.5],
    },
)
fig.axes[0].set_title('Basic heatmap with bins=4')
fig.subplots_adjust(wspace=0.3, hspace=0.5)  # adjust spacing between subplots
# gs = fig.axes[0].get_subplotspec().get_gridspec() # similar effect
# gs.update(wspace=0.3, hspace=0.5)
fig.show()

#%% legend with explicit bin values, and a customized formating of ticks
fig = aa_heatmap(
    corr_df,
    box_kws={
        'sizes': sizes,
        'linewidths': linewidths,
        'background_alpha': 0.5,
        'legend': {
            'bins': [0.1, 0.3, 0.6, 0.9],
            'title': 'Custom bins',
            'label_fmt': '{:.3f}',
            'facecolor': '#aaaaee',
            'edgecolor': '#222222',
        },
    },
    fig=plt.figure(figsize=(10, 8)),
    cmap=cmap,
)
fig.axes[0].set_title('Legend with explicit bin values')
fig.show()

#%% legend with custom labels
fig = aa_heatmap(
    corr_df,
    box_kws={
        'sizes': sizes,
        'background_alpha': 0.3,
        'linewidths': linewidths,
        'legend': {
            'bins': 3,
            'labels': ['Low', 'Medium', 'High'],
            'title': 'Strength',
        },
    },
    fig=plt.figure(figsize=(10, 8)),
    cmap=cmap,
)
fig.axes[0].set_title('Legend with custom labels')
fig.show()

#%% Overlay boxes, on top of clustermap
# Note: arguments (e.g., edgecolors and linewidths) are specified in the ORIGINAL data order 
# (i.e., before clustering/reordering).
clsmap = sns.clustermap(
    corr_df, 
    row_cluster=False,
    cmap=cmap,
    figsize=(8, 8),
)
clsmap.figure.canvas.manager.set_window_title('Clustermap with box overlays')

# 1st overlay: boxes with varying edge colors and linewidths, and a legend
overlay_boxes(
    clsmap,
    sizes=sizes,

    # use None or a fully transparent color (e.g., '#00000000') for no edges
    edgecolors=np.where(corr_df.le(0), '#000000', "#039E6D"),
    
    linewidths=np.linspace(3, 0.1, corr_df.size).reshape(corr_df.shape), # example of varying linewidths
    background_alpha=0.5,
    legend={
        'bins': [0.1, 0.3, 0.6, 0.9],
        'title': 'Correlation',
        'ax': clsmap.ax_heatmap.inset_axes([-0.2, 0, 0.15, 0.3]),  # specify the axis to place the legend
    },
)

# 2nd overlay: additional boxes with full size and black edge color,
# increasing the background alpha to 0.9, and add no additional legend
overlay_boxes(
    clsmap,
    sizes=np.ones_like(corr_df) * 0.98,  # using the percentage of cells expressing the marker as box size
    linewidths=np.full(shape=corr_df.shape, fill_value=1),  # black edge color for all boxes
    facecolors=np.full(shape=corr_df.shape, fill_value='none'),  # transparent fill color for all boxes
    edgecolors=np.full(shape=corr_df.shape, fill_value='#000000'),  # edge color based on marker status
    background_alpha=0.9,
)

# clsmap.figure.show()
plt.show()

# %%
