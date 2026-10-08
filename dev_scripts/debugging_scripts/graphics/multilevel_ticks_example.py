#%% preparation
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from aa_utilities.graphics import draw_multilevel_ticks, overlay_boxes

rng = np.random.default_rng(0)


def make_frame(years=('2023', '2024'), quarters=('Q1', 'Q2', 'Q3'), arms=('Placebo', 'Treated'), doses=('Low', 'High')):
    """A frame with a 3-level MultiIndex on both axes."""
    columns = pd.MultiIndex.from_product([years, quarters, ['alpha', 'beta']], names=['year', 'quarter', 'marker'])
    index = pd.MultiIndex.from_product([arms, doses, ['U', 'V']], names=['arm', 'dose', 'subject'])
    return pd.DataFrame(rng.normal(size=(len(index), len(columns))), index=index, columns=columns)


df = make_frame()

#%% reference: seaborn flattens a MultiIndex into one long label per row/column
fig, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(df, ax=ax, cmap='viridis')
ax.set_title('Reference: plain sns.heatmap with a MultiIndex')

#%% basic usage: pass the index and/or the columns of the plotted frame
# The innermost level stays the ordinary tick labels. Every outer level is drawn beyond it, centred on
# its group, with light separator lines that extend through the levels beneath.
fig, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(df, ax=ax, cmap='viridis')
draw_multilevel_ticks(ax, index=df.index, columns=df.columns)
ax.set_title('Rows and columns')

#%% only one axis: the other one keeps the (flattened) labels seaborn gave it
fig, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(df.droplevel([0, 1]), ax=ax, cmap='viridis')  # rows with a plain index
draw_multilevel_ticks(ax, columns=df.columns)
ax.set_title('Columns only (the rows have a plain index)')

#%% options: rotation of the inner column labels, spacing between levels, text and line styles
fig, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(df, ax=ax, cmap='viridis')
draw_multilevel_ticks(
    ax,
    index=df.index,
    columns=df.columns,
    rotation=45,  # rotation of the innermost column labels (row labels keep their rotation)
    pad=8,  # space between levels, in points
    label_kws={'fontweight': 'bold', 'fontsize': 11, 'color': '#333333'},  # passed to `ax.annotate`
    line_kws={'color': '#555555', 'lw': 1.0, 'ls': '--'},  # default: light gray, thin, solid
)
ax.set_title('rotation, pad, label_kws and line_kws')

#%% restyle or remove afterwards: the drawn artists are returned and stored in `ax.multilevel_ticks`
# `labels[level]` and `separators[level]` follow pandas' level numbers (0 = outermost level).
fig, axes = plt.subplots(1, 2, figsize=(16, 6), layout='constrained')
for ax in axes:
    sns.heatmap(df, ax=ax, cmap='viridis', cbar=False)
ticks = draw_multilevel_ticks(axes[0], index=df.index, columns=df.columns)
assert axes[0].multilevel_ticks is ticks

for label in ticks.columns.labels[0]:  # outermost column level (the years)
    label.set_fontweight('bold')
for line in ticks.columns.separators[0]:  # the lines that bound the years
    line.arrow_patch.set_color('#d62728')
    line.arrow_patch.set_linewidth(1.5)
axes[0].set_title('Restyled through the returned artists')

drawn = draw_multilevel_ticks(axes[1], index=df.index, columns=df.columns)
drawn.index.remove()  # take the row levels off again; `drawn.remove()` removes everything
axes[1].set_title('Row levels removed again')

#%% imshow / matshow: cells are centred on integers, so use `cell_offset=0`
fig, ax = plt.subplots(figsize=(9, 6))
ax.imshow(df.values, cmap='viridis', aspect='auto')
draw_multilevel_ticks(ax, index=df.index, columns=df.columns, cell_offset=0)
ax.set_title('imshow with cell_offset=0')

#%% more levels and uneven group sizes
index = pd.MultiIndex.from_tuples(
    [('A', 'a', 'i', 'x'), ('A', 'a', 'i', 'y'), ('A', 'a', 'ii', 'x'), ('A', 'b', 'i', 'x'),
     ('B', 'a', 'i', 'x'), ('B', 'a', 'i', 'y'), ('B', 'a', 'i', 'z'), ('B', 'b', 'ii', 'x')]
)  # fmt: skip
four_levels = pd.DataFrame(rng.normal(size=(len(index), 6)), index=index, columns=list('uvwxyz'))
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(four_levels, ax=ax, cmap='viridis')
draw_multilevel_ticks(ax, index=four_levels.index)
ax.set_title('Four row levels, uneven groups')

#%% tick labels on the other side: levels are drawn on the side where the inner tick labels are
fig, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(df, ax=ax, cmap='viridis', cbar=False)
ax.xaxis.tick_top()
ax.yaxis.tick_right()
draw_multilevel_ticks(ax, index=df.index, columns=df.columns)
ax.set_title('Tick labels on the top and right')

#%% panels with different width/height ratios: spacing is in points, measured on each panel
fig = plt.figure(figsize=(14, 8), layout='constrained')
grid = fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[1, 2])
panels = [make_frame(), make_frame(years=('2023',), quarters=('Q1', 'Q2')),
          make_frame(doses=('Low', 'Mid', 'High')), make_frame(years=('2023',), quarters=('Q1', 'Q2'), doses=('Low', 'Mid', 'High'))]  # fmt: skip
for cell, panel in zip(grid, panels):
    ax = fig.add_subplot(cell)
    sns.heatmap(panel, ax=ax, cmap='viridis', cbar=False)
    draw_multilevel_ticks(ax, index=panel.index, columns=panel.columns)
fig.suptitle('Panels with different size ratios')

#%% together with overlay_boxes (which accepts the Axes of a plain sns.heatmap)
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df, ax=ax, cmap='viridis')
overlay_boxes(ax, sizes=np.abs(df.values) / np.abs(df.values).max(), edgecolors='#444444', linewidths=0.5, background_alpha=0.4)
draw_multilevel_ticks(ax, index=df.index, columns=df.columns)
ax.set_title('Multi-level ticks and overlay_boxes')

#%% clustermap: pass the heatmap axes and the drawn (clustered) order
# Without clustering the groups stay contiguous. Seaborn puts the row labels on the right, and so do the levels.
grid = sns.clustermap(df, row_cluster=False, col_cluster=False, cmap='viridis', figsize=(10, 7))
draw_multilevel_ticks(grid.ax_heatmap, index=grid.data2d.index, columns=grid.data2d.columns)
grid.figure.suptitle('clustermap without clustering')

#%% clustermap with clustering: groups are consecutive runs in the drawn order,
# so a group that clustering splits up shows up as separate segments. A label longer than its segment
# can overlap its neighbours: use a smaller `fontsize` in `label_kws` then.
grid = sns.clustermap(df, cmap='viridis', figsize=(10, 7))
draw_multilevel_ticks(grid.ax_heatmap, index=grid.data2d.index, columns=grid.data2d.columns)
grid.figure.suptitle('clustermap with clustering')

plt.show()

# %%
