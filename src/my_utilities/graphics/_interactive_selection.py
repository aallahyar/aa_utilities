

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.widgets import LassoSelector


class InteractiveSelection:
    """
    source: https://matplotlib.org/stable/gallery/event_handling/lasso_demo.html
    Select indices from a matplotlib collection (i.e. plotted points) using 
    `LassoSelector`.

    Selected indices are saved in the `selected_indices` attribute. This tool 
    fades out the points that are not part of the selection (i.e., reduces their alpha
    values). If your collection has alpha < 1, this tool will permanently
    alter the alpha values.

    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).

    Parameters
    ----------
    collection : `matplotlib.collections.Collection` subclass
        Collection you want to select from.
    ax : `~matplotlib.axes.Axes`
        Axes to interact with.
    alpha_other : 0 <= float <= 1
        To highlight a selection, this tool sets all selected points to an
        alpha value of 1 and non-selected points to *alpha_other*.
    """

    def __init__(self, collection, ax=None, alpha_other=0.3):
        
        if matplotlib.get_backend() not in ['module://ipympl.backend_nbagg']:
            raise ValueError(
                """Place `%matplotlib widget` in the begining of 
                the running cell to load it properly"""
            )

        if ax is None:
            ax = plt.gca()
        self.ax = ax
        self.canvas = ax.figure.canvas
        self.alpha_other = alpha_other
        
        self.collection = collection
        self.points = collection.get_offsets()
        self.n_points = len(self.points)
        self.selected_indices = []

        # ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.n_points, 1))

        self.lasso = LassoSelector(ax, props=dict(color='black'), onselect=self.on_select)

    def on_select(self, verts):
        path = Path(verts)
        self.selected_indices = np.nonzero(path.contains_points(self.points))[0]
        self.fc[:, -1] = self.alpha_other
        self.fc[self.selected_indices, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

    def disconnect(self):
        self.lasso.disconnect_events()
        # self.fc[:, -1] = 1
        # self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

    def on_click(self, event):
        if event.key == "enter":
            self.disconnect()
            self.ax.set_title(
                f"{len(self.selected_indices)} points are selected\n"
                f"See `selector.selected_indices` for indices"
            )
            self.canvas.draw()
            # print("Selected points:")
            # print(selector.points[selector.selected_indices])


if __name__ == '__main__':
    
    # initialization
    # %matplotlib widget
    import numpy as np
    from matplotlib import pyplot as plt
    from my_utilities.graphics import InteractiveSelection

    rng = np.random.default_rng(seed=42)
    data = rng.uniform(low=0, high=1, size=(100, 2))

    fig = plt.figure(figsize=(5, 5))
    ax = fig.gca()
    
    collection = ax.scatter(data[:, 0], data[:, 1], s=30)
    selector = InteractiveSelection(collection, ax)

    fig.canvas.mpl_connect("key_press_event", func=selector.on_click)
    ax.set_title("Press <Enter> to accept selected points.")

    plt.show()