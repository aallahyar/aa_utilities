import matplotlib.pyplot as plt
import numpy as np

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
y_bases = {pos: values.max() for pos, values in zip(positions, data)}

n_links = 10
x_pairs = [rng.choice(positions, replace=False, size=2) for _ in range(n_links)]
result = links(
    x_left=[pair[0] for pair in x_pairs],
    x_right=[pair[1] for pair in x_pairs],
    text=[f'links {x1}-{x2}' for x1, x2 in x_pairs],
    y_bases=y_bases,
    pad=10,
    ax=ax,
)
print('final y_bases:', result.y_bases)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# no y_bases given: every position starts at the current axis top
ax1.boxplot(x=[np.linspace(1, 100), np.linspace(40, 140)], positions=[0, 1])
# ax1.set_yscale('log', base=10)
n_links = 6
links(
    x_left=[0] * n_links,
    x_right=[1] * n_links,
    text=[f'p-value {i}' for i in range(n_links)],
    ax=ax1,
)

# explicit y_bases: an untouched, tall middle position is still respected
ax2.boxplot(
    x=[np.linspace(1, 100), np.linspace(10, 500), np.linspace(40, 140)],
    positions=[0, 1, 2],
)
ax2.set_yscale('log', base=10)
result = links(
    x_left=[0, 0],
    x_right=[2, 1],
    text=['0 vs 2', '0 vs 1'],
    y_bases={0: 100, 1: 500, 2: 140},
    ax=ax2,
)
print('final y_bases:', result.y_bases)

# order-independence: (1-2) then (1-3), or (1-3) then (1-2), give the same layout
fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(6, 4))
for ax, order in [(ax3, ['A', 'B']), (ax4, ['B', 'A'])]:
    ax.boxplot(x=[np.linspace(1, 10)] * 3, positions=[1, 2, 3])
    pairs = {'A': (1, 2), 'B': (1, 3)}
    result = links(
        x_left=[pairs[name][0] for name in order],
        x_right=[pairs[name][1] for name in order],
        text=order,
        y_bases={1: 10, 2: 5, 3: 15},
        ax=ax,
    )
    ax.set_title(f'input order: {order}')

plt.show()