import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aa_utilities.graphics import links

rng = np.random.default_rng(seed=4210)

fig, ax = plt.subplots(1, 1, figsize=(7, 4))

positions = [0, 1, 2, 5]
data = [
    np.linspace(1, 100, num=100),
    np.linspace(800, 2400, num=100),
    np.linspace(1, 5, num=100),
    np.linspace(100, 150, num=100),
]
ax.boxplot(x=data, positions=positions)
ax.set_yscale('log', base=10)

# seed each position's starting height from its own data max
bases = {pos: values.max() for pos, values in zip(positions, data)}

n_links = 10
x_pairs = [rng.choice(positions, replace=False, size=2) for _ in range(n_links)]
result = links(
    anchors=x_pairs,
    texts=[f'links {x1}-{x2}' for x1, x2 in x_pairs],
    bases=bases,
    colors=['#ff0000', '#aaaaaa'] * (n_links // 2),
    # pad=10,
    ax=ax,
)
print(result)
print('final bases:', result.bases)
print('levels:', result.levels)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# no bases given: every position starts at the current axis top
ax1.boxplot(x=[np.linspace(1, 100), np.linspace(40, 140)], positions=[0, 1])
# ax1.set_yscale('log', base=10)
n_links = 6
links(
    anchors=[(0, 1)] * n_links,
    texts=[f'p-value {i}' for i in range(n_links)],
    ax=ax1,
)

# explicit bases: an untouched, tall middle position is still respected
ax2.boxplot(
    x=[np.linspace(1, 100), np.linspace(10, 500), np.linspace(40, 140)],
    positions=[0, 1, 2],
)
ax2.set_yscale('log', base=10)
result = links(
    anchors=pd.DataFrame({'left': [0, 0], 'right': [2, 1]}),  # a 2-column DataFrame works as well
    texts=['0 vs 2', '0 vs 1'],
    bases=pd.Series({0: 100, 1: 500, 2: 140}),  # ... and so does a Series
    ax=ax2,
)
print('final bases:', result.bases)

# order-independence: (1-2) then (1-3), or (1-3) then (1-2), give the same layout
fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(6, 4))
for ax, order in [(ax3, ['A', 'B']), (ax4, ['B', 'A'])]:
    ax.boxplot(x=[np.linspace(1, 10)] * 3, positions=[1, 2, 3])
    pairs = {'A': (1, 2), 'B': (1, 3)}
    result = links(
        anchors=[pairs[name] for name in order],
        texts=order,
        # bases={1: 10, 2: 5, 3: 15},
        ax=ax,
    )
    ax.set_title(f'input order: {order}')

# horizontal orientation: anchors are y-ticks, brackets grow rightward and expand xlim
fig, ax5 = plt.subplots(figsize=(6, 4))
positions_h = [0, 1, 2]
data_h = [np.linspace(1, 100, num=100), np.linspace(40, 140, num=100), np.linspace(1, 60, num=100)]
ax5.boxplot(x=data_h, positions=positions_h, orientation='horizontal')
result = links(
    anchors=[(0, 1), (1, 2)],
    texts=['p=0.02', 'p=0.03'],
    bases={pos: values.max() for pos, values in zip(positions_h, data_h)},
    orient='h',
    ax=ax5,
)
ax5.set_title("orient='h'")
print('horizontal final bases:', result.bases)

# the result identifies every drawn element, so it can be restyled afterwards
for link in result.links:
    print(link.anchors, 'level', link.level, 'bar at', link.bar_value)
    link.text.set_fontweight('bold')

plt.show()