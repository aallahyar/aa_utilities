
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, colors
import seaborn as sns

from aa_utilities.graphics import heatmap

# data preparation
# rng = np.random.default_rng(seed=42)
# corr_df = pd.DataFrame(rng.uniform(-1, 1, size=(15, 15)))
corr_df = pd.DataFrame(np.arange(-112, 113).reshape(15, 15) / 224)
corr_df.index = corr_df.index.map(lambda i: 'row{:d}'.format(i))
corr_df.columns = corr_df.columns.map(lambda c: 'col{:d}'.format(c))

fig, ax = plt.subplots(figsize=(7, 6))
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
fig.show()

# sanity check:
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(corr_df, cmap=cmap, ax=ax)
fig.show()

plt.show()