
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from aa_utilities.graphics import forest_plot

estimates = pd.DataFrame({
    'estimate': [0.5, 0.60, 0.7, 0.8],
    'conf.low': [0.45, 0.50, 0.65, 0.79],
    'conf.high': [0.55, 0.70, 0.80, 0.91],
})

fig = plt.figure(figsize=(2, len(estimates) / 1.7))
ax = fig.gca()

ax = forest_plot(
    estimates=estimates,
    origin=1,
    line_ys = np.array([1, 0, 3, 2]) + 0.5,
    line_labels=[f'row{i}' for i in range(4)],
    line_sublabels=[f'n={n}' for n in [0, 1, 2, 3]],
    p_values=np.linspace(0, 0.1, 4),
    estimate_labels=estimates.estimate.map(str),
    x_scale=dict(value='log', base=2),
    ax=ax,
)
ax.set_xlim([0.3, 2])
plt.show()