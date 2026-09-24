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
a p-value, above it), stacking them automatically so they don't overlap, and expanding the
y-axis at most once to fit them all. Each argument is an array-like with one entry per link.

**Example**:
```python



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