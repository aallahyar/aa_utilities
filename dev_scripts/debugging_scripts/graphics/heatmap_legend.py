
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, colors

from aa_utilities.graphics import heatmap

# data preparation
corr_df = pd.DataFrame(np.arange(-112, 113).reshape(15, 15) / 224)
corr_df.index = corr_df.index.map(lambda i: 'row{:d}'.format(i))
corr_df.columns = corr_df.columns.map(lambda c: 'col{:d}'.format(c))
cmap = colors.LinearSegmentedColormap.from_list('BlueWhiteRed', ['blue', 'white', 'red'], N=8, gamma=1.0)
sizes = corr_df.abs().values * 0.98 / corr_df.values.max()

#%% legend with bins=int (auto-spaced)
fig, ax = plt.subplots(figsize=(8, 6))
heatmap(
    corr_df,
    cmap=cmap,
    ax=ax,
    cbar_kws={'label': 'Value', 'shrink': 0.5},
    box_kws={
        'sizes': sizes,
        'linewidth': 0.9,
        'background_alpha': 0.7,
        'edgecolors': np.where(corr_df.le(0), '#000000', None),
        'legend': {
            'bins': 4,
            'title': 'Box size',
            'loc': 'upper left',
            'bbox_to_anchor': (1.25, 1.0),
        },
    },
)
ax.set_title('Legend with bins=4 (auto-spaced)')
fig.tight_layout()
fig.show()

#%% legend with bins=array (explicit values)
fig, ax = plt.subplots(figsize=(8, 6))
heatmap(
    corr_df,
    cmap=cmap,
    ax=ax,
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
)
ax.set_title('Legend with explicit bin values')
fig.tight_layout()
fig.show()

#%% legend with custom labels
fig, ax = plt.subplots(figsize=(8, 6))
heatmap(
    corr_df,
    cmap=cmap,
    ax=ax,
    box_kws={
        'sizes': sizes,
        'background_alpha': 0.3,
        'legend': {
            'bins': 3,
            'labels': ['Low', 'Medium', 'High'],
            'title': 'Strength',
            'frameon': False,
        },
    },
)
ax.set_title('Legend with custom labels')
fig.tight_layout()
fig.show()

plt.show()
