import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/Users/kxsj048/Library/CloudStorage/OneDrive-AZCollaboration/projects/auxiliary/aa_utilities/src')
from aa_utilities.helpers.pandas import reorder_by_similarity

# Create a test DataFrame with clear cluster structure
rng = np.random.default_rng(42)
df = pd.DataFrame(
    rng.standard_normal((6, 5)).round(1),
    index=[f'row_{i}' for i in range(6)],
    columns=[f'col_{i}' for i in range(5)],
)
# Inject structure: make rows 2,4 and columns 1,3 similar
df.iloc[[2, 4], [1, 3]] += 15

print("Original DataFrame:")
print(df)
print("\n" + "="*80 + "\n")

# Test 1: Reorder both axes (default)
print("Test 1: Reorder both axes")
result = reorder_by_similarity(df)
print(result)
print('-' * 80 + '\n')

# Test 2: Reorder rows only
print("Test 2: Reorder rows only")
result = reorder_by_similarity(df, axis='rows')
print(result)
print('-' * 80 + '\n')

# Test 3: Reorder columns only
print("Test 3: Reorder columns only")
result = reorder_by_similarity(df, axis='columns')
print(result)
print('-' * 80 + '\n')

# Test 4: Custom row clustering kwargs
print("Test 4: Custom row_kws (ward linkage)")
result = reorder_by_similarity(df, row_kws={'method': 'ward', 'metric': 'euclidean'})
print(result)
print('-' * 80 + '\n')

# Test 5: Custom column clustering kwargs
print("Test 5: Custom col_kws (correlation metric)")
result = reorder_by_similarity(df, col_kws={'metric': 'correlation'})
print(result)
print('-' * 80 + '\n')

# Test 6: Single-row DataFrame (should return copy as-is)
print("Test 6: Single-row DataFrame")
result = reorder_by_similarity(df.iloc[[2], :])
print(result)
print('-' * 80 + '\n')

# Test 7: Single-column DataFrame (should return copy as-is)
print("Test 7: Single-column DataFrame")
result = reorder_by_similarity(df.iloc[:, [1]])
print(result)
print('-' * 80 + '\n')

# Test 8: Non-numeric column should raise TypeError
print("Test 8: Non-numeric column (expect TypeError)")
try:
    bad_df = df.copy()
    bad_df['str_col'] = 'x'
    reorder_by_similarity(bad_df)
except TypeError as e:
    print(f"  Caught: {e}")
print('-' * 80 + '\n')

# Test 9: NaN values should raise AssertionError
print("Test 9: NaN values (expect AssertionError)")
try:
    bad_df = df.copy()
    bad_df.iloc[0, 0] = np.nan
    reorder_by_similarity(bad_df)
except AssertionError as e:
    print(f"  Caught: {e}")
print('-' * 80 + '\n')

# Test 10: Invalid axis should raise ValueError
print("Test 10: Invalid axis (expect ValueError)")
try:
    reorder_by_similarity(df, axis='invalid')
except ValueError as e:
    print(f"  Caught: {e}")
print('-' * 80 + '\n')

# Test 11: row_kws with axis='columns' should raise ValueError
print("Test 11: row_kws with axis='columns' (expect ValueError)")
try:
    reorder_by_similarity(df, axis='columns', row_kws={'method': 'ward'})
except ValueError as e:
    print(f"  Caught: {e}")
print('-' * 80 + '\n')

# Test 12: col_kws with axis='rows' should raise ValueError
print("Test 12: col_kws with axis='rows' (expect ValueError)")
try:
    reorder_by_similarity(df, axis='rows', col_kws={'method': 'ward'})
except ValueError as e:
    print(f"  Caught: {e}")
print('-' * 80 + '\n')
