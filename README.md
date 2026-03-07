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

#### `add_pvalue`:
Adds a link and a string (e.g., p-value) over a given pair of `x_ticks`.

**Example**:
```python

from aa_utilities.graphics import add_pvalue

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