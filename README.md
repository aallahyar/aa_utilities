# My utilities
A general purpose package that contains series of simple scripts used in my day-to-day work.

The file structure is inspired from [this RealPython tutorial](https://realpython.com/pypi-publish-python-package/).

## Installation

You can download the repository install the package using:
```bash
cd my_utilities

# editable installtion
python3 -m pip install --editable .
# current path (i.e. `./`) should be containing the `pyproject.toml`` file.
```

## How to use

#### `add_pvalue`:
Example:
```python
from my_utilities import add_pvalue

df = pd.DataFrame({
    'a': range(100),
    'b': np.arange(40, 140),
})
print(df)

fig = plt.figure()
ax = fig.gca()
sns.boxplot(
    data=df.melt(var_name='col', value_name='val'),
    x='col',
    y='val',
    ax=ax,
)
add_pvalue(ax, 'test', [0, 1.2], 130, 170, 150)
plt.show()
```