
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

cmap = colors.LinearSegmentedColormap.from_list('BlueWhiteRed', ['blue', 'white', 'red'], N=8, gamma=1.0)
fig = heatmap(
    corr_df,
    box_kws={
        'background_alpha': 0.7,
        'sizes': corr_df.abs().values * 0.98 / corr_df.values.max(),
        'linewidth': 0.9,
        'edgecolors': np.where(corr_df.le(0), '#000000', None),
        'legend': {
            'bins': 4,
            'title': 'Box size',
        },
    },
    figsize=(10, 8),
    cmap=cmap,
    cbar_kws={
        'label': 'Label of the colorbar',
        'extend': 'max',
        'ticks': [-4, 0, 4],
    },
)
fig.show()

# sanity check:
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(corr_df, cmap=cmap, ax=ax)
fig.show()

plt.show()