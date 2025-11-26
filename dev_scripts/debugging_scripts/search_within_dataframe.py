import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/Users/kxsj048/Library/CloudStorage/OneDrive-AZCollaboration/projects/auxiliary/aa_utilities/src')
from aa_utilities.helpers.pandas import search

# Create a comprehensive test DataFrame
df = pd.DataFrame({
    'id': [1, 2, 3, 4, 5, 6, 7, 8],
    'value_int': [10, 20, 10, 30, 10, 40, 50, 10],
    'value_float': [1.5, 2.5, 1.5, 3.5, np.nan, 1.501, 4.5, np.inf],
    'name': ['Alice', 'Bob', 'alice', 'Charlie', 'Alice', 'David', 'Eve', 'Alice'],
    'city': ['NYC', 'LA', 'NYC', 'Chicago', 'nyc', 'Boston', 'NYC', 'Seattle'],
    'mixed': ['test', 100, 'test', 200, 'Test', 300, 'test', 400]
}, index=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])

print("Test DataFrame:")
print(df)
print("\n" + "="*80 + "\n")

# Test 1: Find exact integer value
print("Test 1: Find exact integer 10")
result = search(df, 10)
print(result)
print()

# Test 2: Find exact float value
print("Test 2: Find exact float 1.5")
result = search(df, 1.5)
print(result)
print()

# Test 3: Find float with tolerance
print("Test 3: Find float ~1.5 with tolerance=0.01")
result = search(df, 1.5, tolerance=0.01)
print(result)
print()

# Test 4: Find NaN
print("Test 4: Find NaN")
result = search(df, np.nan)
print(result)
print()

# Test 5: Find exact string (case-sensitive)
print("Test 5: Find 'Alice' (case-sensitive)")
result = search(df, 'Alice', case=True)
print(result)
print()

# Test 6: Find string (case-insensitive)
print("Test 6: Find 'alice' (case-insensitive)")
result = search(df, 'alice', case=False)
print(result)
print()

# Test 7: Find string with regex
print("Test 7: Find pattern 'A.*' (regex)")
result = search(df, 'A.*', regex=True)
print(result)
print()

# Test 8: Find 'NYC' (case-sensitive)
print("Test 8: Find 'NYC' (case-sensitive)")
result = search(df, 'NYC', case=True)
print(result)
print()

# Test 9: Find 'nyc' (case-insensitive)
print("Test 9: Find 'nyc' (case-insensitive)")
result = search(df, 'nyc', case=False)
print(result)
print()

# Test 10: Find value that doesn't exist
print("Test 10: Find non-existent value 999")
result = search(df, 999)
print(result)
print("(Should be empty)")

# Test 11: Find value in mixed type column
print("Test 11: Find 'test' in mixed type column")
result = search(df, 'test', case=True)
print(result)
print()

# Test 12: Find value in mixed type column
print("Test 12: Find 100 in mixed type column")
result = search(df, 100, case=True)
print(result)
print()

