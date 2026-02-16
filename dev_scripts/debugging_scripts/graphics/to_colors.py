import numpy as np

from aa_utilities.graphics import to_colors

# Linear scale
colors = to_colors([1, 2, 3, 4, 5, np.nan, None], cmap="viridis", default='#888888')
print(colors)
    
# Log scale
colors = to_colors([1, 10, 100, 1000], scale="log", cmap="plasma")
print(colors)

# Diverging scale centered at zero
colors = to_colors([-5, -2, 0, 2, 5], scale="diverging", vcenter=0, cmap="RdBu")
print(colors)

# Categorical strings
colors = to_colors(['apple', 'banana', 'apple', 'cherry', 'banana', None], cmap="Set3", default="#888888")
# Returns 6 colors (including gray for NaN): 
# apple and the 3rd value get the same color, banana and the 5th get the same color
print(colors)
