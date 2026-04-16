
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, colors
import seaborn as sns

from aa_utilities.graphics import heatmap

# data preparation
corr_df = pd.DataFrame(np.arange(-112, 113).reshape(15, 15) / 224)
corr_df.index = corr_df.index.map(lambda i: 'row{:d}'.format(i))
corr_df.columns = corr_df.columns.map(lambda c: 'col{:d}'.format(c))
cmap = colors.LinearSegmentedColormap.from_list(
    'BlueWhiteRed', ['blue', 'white', 'red'], N=8, gamma=1.0)
sizes = corr_df.abs().values * 0.98 / corr_df.values.max()

#%% basic heatmap with boxes + legend
fig = heatmap(
    corr_df,
    box_kws={
        'background_alpha': 0.7,
        'sizes': sizes,
        'linewidth': 0.9,
        'edgecolors': np.where(corr_df.le(0), '#000000', None),
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
        'ticks': [-4, 0, 4],
    },
)
fig.axes[0].set_title('Basic heatmap with bins=4')
fig.subplots_adjust(wspace=0.3, hspace=0.5)  # adjust spacing between subplots
# gs = fig.axes[0].get_subplotspec().get_gridspec() # similar effect
# gs.update(wspace=0.3, hspace=0.5)
fig.show()

#%% legend with explicit bin values
fig = heatmap(
    corr_df,
    box_kws={
        'sizes': sizes,
        'linewidth': 0.9,
        'background_alpha': 0.5,
        'legend': {
            'bins': [0.1, 0.3, 0.6, 0.9],
            'title': 'Custom bins',
            'label_fmt': '{:.1f}',
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
fig = heatmap(
    corr_df,
    box_kws={
        'sizes': sizes,
        'background_alpha': 0.3,
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

#%% sanity check: plain sns.heatmap
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(corr_df, cmap=cmap, ax=ax)
ax.set_title('Sanity check: plain sns.heatmap')
fig.show()

plt.show()