from aa_utilities.graphics import to_colors

# Linear scale
colors = to_colors([1, 2, 3, 4, 5], cmap_name="viridis")
print(colors)
    
# Log scale
colors = to_colors([1, 10, 100, 1000], scale="log", cmap_name="plasma")
print(colors)

# Diverging scale centered at zero
colors = to_colors([-5, -2, 0, 2, 5], scale="diverging", vcenter=0, cmap_name="RdBu")
print(colors)

