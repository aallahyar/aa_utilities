
## [0.0.10] - 2024-XX-XX
### Changed:
- `R['var']` now returns a scalar, if the `var` is atomic, and also has `length(var)=1`

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