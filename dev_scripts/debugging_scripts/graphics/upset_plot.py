
from aa_utilities.graphics import UpsetPlot

# ── sample data ──────────────────────────────────────────────────────────
sets = {
    'Gene Set A': {1, 2, 3, 4, 5, 6, 7},
    'Gene Set B': {4, 5, 6, 7, 8, 9, 10},
    'Gene Set C': {6, 7, 8, 9, 10, 11, 12, 13},
    'Gene Set D': {1, 10, 11, 14, 15},
}

# ── 1. Basic inclusive plot ──────────────────────────────────────────────
up = UpsetPlot(sets)
print('=== intersections (inclusive) ===')
print(up.intersections)
up.plot(show=True)

# ── 2. Exclusive mode ───────────────────────────────────────────────────
up_ex = UpsetPlot(sets, mode='exclusive')
print('\n=== intersections (exclusive) ===')
print(up_ex.intersections)
up_ex.plot(col_stripe="#7dc1f1", colors={'excluded': '#cc3333'})

# ── 3. Filter: keep only intersections with size >= 1, sorted by size ──
up_filtered = (
    UpsetPlot(sets, mode='exclusive')
    .filter(lambda ix: ix[ix['_size'] >= 1].sort_values('_size', ascending=False))
)
print('\n=== filtered (exclusive, size >= 1) ===')
print(up_filtered.intersections)
up_filtered.plot(row_stripe="#efe0a2", col_stripe='#f0f0ff')

# ── 4. Filter: drop a set column ────────────────────────────────────────
def drop_set_d(df):
    df = df.drop(columns=['Gene Set D'])
    df[df == 0] = -1  # make exclusive
    return df[df['_size'] >= 1]

up_drop = (
    UpsetPlot(sets)
    .filter(drop_set_d)
)
print('\n=== after dropping Gene Set D ===')
print(up_drop.intersections)
up_drop.plot()

# ── 5. Deferred render + artist customisation ───────────────────────────
up_custom = (
    UpsetPlot(sets, mode='exclusive')
    .filter(lambda df: df[df['_size'] >= 1])
    .plot(show=False)
)
up_custom.ax_intersection_sizes.set_ylabel('Count')
up_custom.ax_intersection_sizes.set_title('My UpSet Plot')
up_custom.show()

# --- 6. sort sets by size ───────────────────────────────────────────────────────
up_sorted = UpsetPlot(sets)
sorted_cols = up_sorted.sets.sort_values('size', ascending=False).index.tolist()
up_sorted.filter(lambda ix: ix[sorted_cols])
up_sorted.plot()
