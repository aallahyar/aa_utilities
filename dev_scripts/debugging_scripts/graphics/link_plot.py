import matplotlib.pyplot as plt
import numpy as np

from aa_utilities.graphics import links

rng = np.random.default_rng(seed=42)

fig, ax = plt.subplots(1, 1, figsize=(7, 4))

ax.boxplot(
    x=[
        np.linspace(1, 100, num=100), 
        np.linspace(40, 140, num=100),
        np.linspace(10, 140, num=100),
    ], 
    positions=[0, 1, 3],
)
ax.set_yscale('log', base=10)
# ax.set_ylim(top=1e10)

n_links = 10
x_pairs = [rng.choice([0, 1, 3], replace=False, size=2) for _ in range(n_links)]
links(
    x_left=[pair[0] for pair in x_pairs],
    x_right=[pair[1] for pair in x_pairs],
    text=[f'test p-value = string {i}' for i in range(n_links)],
    y_left=[100] * n_links,
    y_right=[140] * n_links,
    ax=ax,
)
plt.show()
