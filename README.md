# AA utilities
A general purpose package that contains series of simple scripts used in my day-to-day work.

The file structure is inspired from [this RealPython tutorial](https://realpython.com/pypi-publish-python-package/).

## Installation

### 1. Manual download and installation:
You can download the repository and install the package using:
```bash
git clone https://github.com/aallahyar/aa_utilities.git
cd aa_utilities

# editable installtion
python3 -m pip install --editable .
# current path (i.e. `./`) should be containing the `pyproject.toml`` file.
```

### 2. Installing via `pip install`:
You can also install using `pip`, via:
```bash
pip3 install git+https://github.com/aallahyar/aa_utilities
```

### 3. Adding the packge to `requirements.txt`:
Alternatively, you can install it using `pip`s specification in `requirement.txt` file as follows:
```
# in ./requirements.txt
# 1. For installing the latest commit:
aa_utilities @ git+https://github.com/aallahyar/aa_utilities

# 2. For installing a specific version/tag
aa_utilities @ git+https://github.com/aallahyar/aa_utilities.git@v0.0.3
```
This installs version `0.0.3` of the package.
The further details about this is explained [here](https://stackoverflow.com/questions/16584552/how-to-state-in-requirements-txt-a-direct-github-source).

## How to use

#### `links`:
Draws one or more comparison "links" (a bracket connecting two x-positions with a text, e.g.
a p-value, above it), automatically packing overlapping ones onto separate vertical levels so
a link's bar always clears any position/link it spans over, and expanding the y-axis at most
once to fit them all. `x_left`, `x_right`, and `text` are array-likes with one entry per link.
`y_bases` is an optional mapping from x-position to its real starting height - positions you
don't list default to the current axis top. The level packing (and its results) don't depend
on the order links are given in; pass `auto_order=False` to assign levels using the given
input order instead of the narrowest-span-first packing.

**Example**:
```python
from aa_utilities.graphics import links

fig = plt.figure()
ax = fig.gca()
ax.boxplot(x=[range(100), range(40, 140)], positions=[0, 1])
result = links(
    x_left=[0],
    x_right=[1],
    text=['test p-value = string'],
    y_bases={0: 130, 1: 150},
    ax=ax,
)
plt.show()
```

## Running tests
```bash
# run all tests
# `pyproject.toml` already sets testpaths = ["tests"], so bare pytest knows where to look.
python3 -m pytest

# run only the RSpace tests
python3 -m pytest tests/wrappers/test_rspace.py

# with verbose output (shows each test name)
python3 -m pytest tests/wrappers/test_rspace.py -v

# stop on first failure
python3 -m pytest -x

# run a single test by name
python3 -m pytest tests/wrappers/test_rspace.py::test_named_list_of_dataframes_from_r
```