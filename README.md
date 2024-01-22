# My utilities
A general purpose package that contains series of simple scripts used in my day-to-day work.

The file structure is inspired from [this RealPython tutorial](https://realpython.com/pypi-publish-python-package/).

## Installation

You can download the repository and install the package using:
```bash
git clone https://github.com/aallahyar/my_utilities.git
cd my_utilities

# editable installtion
python3 -m pip install --editable .
# current path (i.e. `./`) should be containing the `pyproject.toml`` file.
```
Alternatively, you can install it using `pip`s specification in `requirement.txt` file as follows:
```
# in requirements.txt
my_utilities @ git+https://github.com/aallahyar/my_utilities.git@v0.0.3
```
This installs version `0.0.3` of the package

## How to use

#### `add_pvalue`:
Adds a link and a string (e.g., p-value) over a given pair of `x_ticks`.

**Example**:
```python

from my_utilities.graphics import add_pvalue

fig = plt.figure()
ax = fig.gca()
ax.boxplot(x=[range(100), range(40, 140)], positions=[0, 1])
add_pvalue(
    x_ticks=[0, 1], 
    text='test p-value = string', 
    y_left=130, 
    y_top=170, 
    y_right=150, 
    ax=ax,
)
plt.show()
```