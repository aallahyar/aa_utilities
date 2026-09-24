import matplotlib.pyplot as plt
import numpy as np

from aa_utilities.graphics import links

rng = np.random.default_rng(seed=42)

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
    text=[f'test p-value = string {i}' for i in range(n_links)],
    y_bases=y_bases,
    # pad=10,
    ax=ax,
)
print('final y_bases:', result.y_bases)
plt.show()

