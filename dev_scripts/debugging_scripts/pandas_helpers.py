import numpy as np
import pandas as pd

from aa_utilities.helpers.pandas import describe

# Sample DataFrame
data = {
    "age"    : [25, 30, 28, 35, 40, np.nan, np.nan],
    "salary" : [50000.0, 60000.0, 55000.0, 70000.0, 80000.0, 0, np.nan],
    "bonus"  : [5000, 6000, 5500, 7000, 8000, 9000, 10000],
    "name"   : ["Alice", "Bob", "Charlie", "David", "Eve", np.nan, np.nan],
    "city"   : pd.Categorical(["New York", "Los Angeles", "Chicago", "Houston", "New York", "Los Angeles", "Chicago"]),
}
df = pd.DataFrame(data)

# 1 — Default: both tables shown, top 3 elements
print("=== Default: Both Numeric and Categorical Summary ===")
describe(df)

# 2 — Only numeric table
print("=== Only Numeric Summary ===")
describe(df, show_categorical=False)

# 3 — Only categorical table
print("=== Only Categorical Summary ===")
describe(df, show_numeric=False)

# 4 — Categorical only, with top 5 most frequent elements
print("=== Categorical Summary with Top 5 Elements ===")
describe(df, n_top=5, show_numeric=False)

# 5 — Capture outputs without displaying
print("=== Capture Outputs Without Displaying ===")
num_df, cat_df = describe(df, show_numeric=False, show_categorical=False)