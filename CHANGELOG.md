## [0.0.24] - 2025-XX-XX
### Added:
- `graphics.add_counts_to_legend` to add counts to the legend of a given ax.

## [0.0.23] - 2024-11-20
### Added:
- `convenience.Logger` which can restrict logs printed by a logger based on number of logs printed or the time since last print.

## [0.0.22] - 2024-11-15
### Added:
- When printed, `graphics.InteractiveSelection` object now show some information about what it holds.
- `convenience.store` to store a given `data` into a given container with a given `name`.
This is useful during the `Pandas` chaining to catch a state of the chained dataframe and store it
in another variable.

## [0.0.21] - 2024-10-07
### Added:
- `select` to select subsets of rows from a `DataFrame` according to a given dictionary 
of `queries`.
### Fixed:
- `RSpace` is now retuning a `pd.Series` or `pd.DataFrame` depending on the dimension of
the requested variable.

## [0.0.20] - 2024-10-01
### Changed:
- `Dataframes` should now be properly named when moved out of `R`.

### Changed:
- `convenience.LinkageTreeParser` now produces a less verbose output.

## [0.0.19] - 2024-09-02
### Added:
- `get_logger` now has an additional attribute that stores the last time a message was printed.

### Changed:
- `convenience.is_true` now receives the `dataframe` as its first argument.
- `convenience.is_true` now can receive a function as the `condition` which will be 
called on the given `dataframe`. The function needs to return `True` for a successfull function call.

### Fixed:
- `get_logger` now always return the logger, even if its already initiated.

## [0.0.18] - 2024-08-28
### Added:
- `convenience.get_logger` to create a local logger with specific `name` and logging `level`.
- `convenience.is_true` to test a given `condition` and return the provided data `data` if 
condition is `condition=True`. Useful for `Pandas` chaining and testing conditions.

## [0.0.17] - 2024-07-26
### Added:
- `convenience.LinkageTreeParser` to parse scikit-learn's `AgglomerativeClustering` trees.

## [0.0.16] - 2024-07-21
### Added:
- `wrappers.RSpace` is now returning `numpy.ndarrays` if value is a list of `string`.
- `convenience.generate_dataframe()` is now added to generate dummy dataframes 
with specific `n` (i.e., number of samples) and `seed`.
- `InteractiveSelection` is added to allow interactive seleciton of points in a 
`scatterplot`.

## [0.0.15] - 2024-06-20
### Changed:
- `wrappers.RSpace` is now choosing to return `numpy.ndarrays` (instead of `pandas.DataFrame`) if the 
array dimension is more than 2D.

## [0.0.14] - 2024-06-17
### Changed:
- `forest_plot` now has a `line_sublabel` argument to show lines below each line.
- `forest_plot` no longer has `counts` argument (removed in favor of `line_sublabel` argument).

## [0.0.13] - 2024-06-08
### Added:
- `pvalue_to_asterisks` function, which converts `p_values` to astericks 
(e.g., `0.002` ==> `**`)

### Fixed:
- A bug in `link`, where its `text_kw` were not set.
- A bug in `link`, where the link was not drawing properly in log scaled axes.
- A bug in `forest_plot`, where labels were differently distanced based on how large
the figure was.

## [0.0.12] - 2024-05-20
### Added:
- `interval2str` function is added to convert `Pandas.Interval` data types to a more 
human-readable string.
### Changed:
- `forest_plot` now has a `line_ys` argument to determine the location of each estimate

## [0.0.11] - 2024-05-15
### Added:
- `forest_plot` is added to produce `ForestPlots`, useful in treatment-response presentations.

### Changed:
- `add_pvalue` function is now called `link` to make it more intuitive.

## [0.0.10] - 2024-05-05
### Changed:
- `R['var']` now returns a scalar, if the `var` is atomic, and also has `length(var)=1`
- `add_pvalue` function is now called `link` to make it more intuitive.

## [0.0.9] - 2024-04-22
### Fixed:
- A bug in `RSpace`, where its `IPython` extension was loaded with a wrong syntax.
### Changed:
- Expanded installation explanations in `README.md`.

## [0.0.8] - 2024-03-05
### Changed:
- Added an `ipython` flag to `RSpace` constructor to initiate 
[its extension](https://rpy2.github.io/doc/latest/html/interactive.html#usage).
- Removed strict version definitions in `requirements.txt`. E.g., `natsort==8.4.0`
to `natsort>=8.4.0`


## [0.0.7] - 2024-03-05
### Added:
- Added a helper function to count the number of elements shared between a given 
    set of lists or iterables.

## [0.0.6] - 2024-03-05
### Added:
- Added a helper function to perform multi-column natrual sorting.


## [v0.0.5] - 2024-02-20
### Added:
- `RSpace`: Now returns a `Pandas.DataFrame` (including a `column`/`index`) when 
the variable contains `name()`, `rownames()` or `colnames()`


## [v0.0.4] - 2024-01-25
### Removed:
- Minor clean up including removing redundant files.


## [v0.0.3] - 2024-01-24
### Changed:
- Restructured the code to hide private packages from user and auto-complete.
- Added a conditional import for packages that don't have their prerequisite packages 
installed (e.g., `RSpace` class that requires `rpy2`).


## [v0.0.2] - 2024-01-22
### Changed:
- Added `config.toml` file as resource.


## [v0.0.1] - 2024-01-22
First release of the utility scripts.

# template: https://keepachangelog.com/en/1.0.0/