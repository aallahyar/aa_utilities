
## [0.0.59] - 2026-11-24
Fixes from an independent review of the whole package.
### Added:
- `LinearModel` now tracks where its R objects come from. `data`, `fit`, `emmeans` and `contrasts` each get a token, kept in `LinearModel.lineage` and in a hidden R-side registry (`.__aa_token`) that also holds a reference to each registered R object. `fit_*()`, `add_emmeans()` and `add_contrasts()` check that the R objects they need (and everything upstream) are still the ones this model created, and raise the new `StaleResultError` otherwise: after `set_data()`/`set_reference()`, after a failed refit, when another `LinearModel` sharing the R session re-created `fit`, or when `fit`/`LSmeans`/`data` was re-assigned or edited by hand in R. Editing `data` by hand *before* fitting is adopted with a warning. A model only removes registry entries it owns, so clearing one model does not affect another sharing the R session, and a variable deleted in R (e.g., `rm(data)`) is refused instead of resolving to an R function of the same name.
### Changed:
- **Breaking:** `LinearModel.fit_*()` need `set_data()` first: data assigned directly in R is refused. `LinearModel()` without `space` now creates its own `RSpace` (it used to crash).
- **Breaking:** `computation.modeling.remove_effects` matches `covs_remove`/`covs_keep` to design-matrix terms exactly (through patsy's term map), instead of by name prefix: removing `age` no longer also removes `age_baseline`, and no longer removes interactions such as `age:sex` unless they are listed.
- `helpers.formatters.pvalue_to_asterisks` returns `'NA'` for a missing p-value (it was `'ns'`). `stats.IterativeStats.mean()` is `nan` before any value is included (it was `0.0`).
- `computation.pairwise_associations` tests boolean columns as categorical (chi-square/Kruskal-Wallis), as intended, instead of as numeric.
- `RSpace` no longer mutes warnings when R runs with `options(warn = 2)`, so they become errors as in plain R.
- `helpers.pandas.describe(return_dfs=True)` works with the default flags and also summarizes `string` columns.
- `LinearModel` confidence levels are no longer rounded to 2 decimals (`ci=0.954` used to become `0.95`), and must be between 0 and 1.
- `storage.Checkpoint.save()` writes to a uniquely named temporary file and replaces the target only on success. `overwrite=False` is now atomic.
- Requirements: `scipy`, `joblib` and `patsy` are declared as dependencies (they were imported but only available through other packages or extras). CI installs the R packages used by `LinearModel` (and fails if one cannot be installed), and R-dependent tests fail instead of skipping on CI, also when `RSpace` cannot be imported.
### Fixed:
- `computation.pairwise_distances` gave different results for `seuclidean` and `mahalanobis` depending on `n_jobs`, as every chunk estimated its own variance/covariance (also for the aliases `se`, `s`, `mah`, `mahal` and an explicit `V=None`/`VI=None`; the defaults are estimated in float64 like scipy does). An empty `X` now works for any `n_jobs`.
- `storage.Container.copy/update/drop/filter/from_series` lost keys starting with `_` and failed on non-string keys.
- `helpers.pandas.sort` duplicated rows when the index had duplicates. `search` reported `col_number` relative to the searched columns, failed on nullable columns and on numeric object columns for case-insensitive/regex searches. `select` with no queries and `reorder_by_similarity` with nullable integers now behave.
- `computation.pairwise_associations` crashed on categorical columns with missing values.
- `computation.modeling.StabilitySelector` replaced a failure reported by the selector with `None`, and aborted when the estimator factory raised.
- The Spearman correlation in `computation._corr` ignored ties (and ranked NaN). Ties get their average rank and NaN is rejected.
- `LinearModel.add_contrasts` gave repeated contrast labels (e.g., the same comparison in several visits) the context of the last one. `clear_results(fit=True)` also clears `n_subjects` and `fit_theta`.
- `LinearModel.get_model_formula()` returns the cached formula and is refused when the R `fit` is not this model's. `LinearModel.add_contrasts` left the `operator` column empty when the contrasts are conditioned on another predictor (`spec='group | visit'`), and failed for numeric predictors (their levels now stay numeric in the `_left`/`_right` columns). `factorize()` requires data set by this model, a `set_reference()` that fails halfway invalidates the fit it made stale, and `fit_logistic` validates `ci` before discarding the previous fit.
- `RSpace`: zero-column matrices keep their rows, non-atomic R objects (functions, environments) are no longer turned into a `Series`, `clear_warnings()`/`clear_messages()` inside a `capture_*()` block no longer loses later entries, and `close()` works without a running IPython shell.
- `graphics.UpsetPlot` counted duplicates in set sizes, failed on generators, dropped distinct intersections whose labels collided, and accepted set names that clash with its metadata columns (`_size`, ...). Empty input is rejected.
- `graphics.links` assigned levels depending on the input order for equally wide links.
- `graphics.overlay_shapes` derived automatic legend bins from masked or NaN cells too.
- `graphics.jitter` failed for lists and arrays. `graphics.to_colors` treated numbers mixed with `None` as categories, and turned missing strings into the text `'nan'`.
- `graphics.repel_text` rejects non-finite anchors (it crashed). `graphics.InteractiveSelection` uses the axes of its collection by default.
- `loggers.RestrictedLogger` ignored `setLevel()` after construction. `computation.LinkageTreeParser` explains how to fit a scikit-learn model that lacks `distances_`.
- `helpers.watermark` could combine the date of one day with the time of another. Corrected examples in the docstrings of `is_true`, `select`, `watermark` and `TextWrapper`, and in the README.

## [0.0.58] - 2026-11-19
### Changed:
- **Breaking:** `graphics.overlay_boxes` is renamed to `graphics.overlay_shapes`, as it draws shapes and not only boxes. The axes of its size legend are now labelled `size_legend` (was `marker_legend`). Its docstring has an example that lays out the heatmap, the colorbar and the size legend with a `GridSpec` and passes the legend axes through `legend={'ax': ...}`.
### Removed:
- **Breaking:** `graphics.heatmap` is removed. `sns.heatmap` on a `GridSpec` followed by `overlay_shapes(..., legend={'ax': legend_ax})` produces the same figure (pixel-identical), and also supports `facecolors` and clustermaps. See the example in `overlay_shapes`.

## [0.0.57] - 2026-11-18
### Added:
- `graphics.overlay_boxes` (and `graphics.heatmap`'s `box_kws`) accept `shapes`: a single matplotlib marker code, or per-cell codes in the original data order (like `sizes`), to draw shapes other than squares. Supported: `s o D d ^ v < > p h H 8 * P X` and `?` (a question mark). Every shape is scaled so that its larger bounding-box side equals the cell's `size`, and `^`/`v`/`<`/`>` point the way they look on screen. Unfilled markers such as `+` and `x` are rejected; use `P` and `X`. The default is `'s'`, so existing calls look the same. The size legend draws the shape shared by all cells (squares if they are mixed); pass `legend={'shape': <code>}` to choose another.
### Changed:
- The boxes drawn by `graphics.overlay_boxes`/`graphics.heatmap` and by their size legend are now `PathPatch`es instead of `Rectangle`s (same geometry and rendering for squares).

## [0.0.56] - 2026-11-11
### Fixed:
- `graphics.overlay_boxes`/`graphics.heatmap`'s marker-range legend (`_draw_box_legend`) no longer draws its largest box straddling the legend axes' left edge (previously `clip_on=False` let roughly half of it bleed into the heatmap/colorbar to its left); all legend boxes are now right-aligned flush within the legend axes.

## [0.0.55] - 2026-10-17
### Changed:
- **Breaking:** `helpers.plotly.build_sankey_data` now returns `(link, node)` instead of `(node, link)`, matching its `(link_df, node_df)` argument order.
- `RSpace` no longer captures warnings/errors by scraping R's console text (which could split one multi-line warning into several entries, and conflated warnings with errors). It now runs every snippet through a hidden R closure that uses `withCallingHandlers`/`tryCatch` to capture each condition as structured data (full, untruncated `message`, `call`, and `classes`) - repeated identical warnings are still each recorded individually. This is a transparent change: the snippet still runs via `eval(parse(text=...), envir=globalenv())` exactly like plain `ro.r()`, including its invisible-result semantics (e.g. `x <- 1` still returns `None`), and the actual computed values are unaffected (verified bit-for-bit against direct `rpy2` calls).
- An R error now raises `RError` (a `RuntimeError` subclass carrying `message`/`call`/`classes`) instead of a generic `RuntimeError` built from `rpy2`'s flat error string. Any warnings/messages raised before the error are still recorded in `self.warnings`/`self.messages`.
### Added:
- `RSpace` now captures R `message()` calls separately in `self.messages` (as `RMessage`), with matching `clear_messages()`/`capture_messages()`.

## [0.0.54] - 2026-10-16
### Added:
- `graphics.links` now accepts `orient='vertical'|'horizontal'` (or `'v'|'h'`) to draw comparison links for horizontal boxplots/bars as well (bars grow rightward and expand the x-axis instead of the y-axis).
- `graphics.links` returns more about what it placed: `LinkArtists` (now exported from `graphics`) carries `anchors`, `level`, `feet`, `bar_value` and `bar_top` next to the `line`/`text` artists, and `LinksResult` has `levels` and `orient`, so drawn elements can be identified and adjusted afterwards.
- `graphics.links` accepts a 2-column DataFrame as `anchors` and a `pd.Series` as `bases`.
- `helpers.plotly.build_sankey_data` forwards every other column of `link_df`/`node_df` to Plotly under its own name (e.g. link `label`/`customdata`, node `hovertemplate`), and clips node `x`/`y` into (0, 1) as Plotly rejects exactly 0 or 1. `node_df` is optional: nodes are then inferred from `link_df` (sources first, then targets, by first appearance).
### Changed:
- **Breaking:** `helpers.plotly.build_sankey_data` now takes `(link_df, node_df=None)` (arguments swapped), no longer requires `label`, and no longer ignores unknown columns: Plotly rejects names it does not know, so select the columns to pass.
- **Breaking:** `graphics.links` arguments `x_lefts`/`x_rights` are replaced by a single `anchors` `(n, 2)` array-like, and `y_bases` is renamed to `bases`, as they are no longer always x/y. Everything after `texts` is keyword-only. `LinksResult.y_bases` is now `LinksResult.bases`, and `LinkArtists` is a dataclass with `y_top` renamed to `bar_value` (no longer unpackable as a tuple).

## [0.0.53] - 2026-10-08
### Added:
- `graphics.draw_multilevel_ticks` draws the levels of a `MultiIndex` on the rows/columns of a heatmap (e.g., `sns.heatmap`, or a clustermap's heatmap), on the side where the tick labels are, instead of seaborn's flattened labels. It returns (and stores in `ax.multilevel_ticks`) the drawn texts and lines per level, so they can be restyled or removed.
### Fixed:
- `RSpace` now converts length-1 logical (`TRUE`/`FALSE`/`NA`) and complex R values to Python scalars. Previously a returned `FALSE` was an rpy2 vector that evaluated as truthy.
- `RSpace` now returns `NA` in R integer/logical vectors and data.frame columns as pandas nullable `Int64`/`boolean` values (and a lone `NA` as `pd.NA`), instead of the integer sentinel `-2147483648` or an rpy2 NA object.
- `RSpace` now keeps the shape of logical, character and complex matrices (they used to become a single column, and named logical matrices raised an error), and also applies the `NA` handling above to matrices. Arrays with more than two dimensions are numpy arrays, and 1-D arrays (e.g. `table(x)`) are a `pd.Series`.
- `LinearModel.add_contrasts` no longer fails for models with an `offset()` (e.g., `fit_negbin`): `broom` returns `null.value` as a matrix column for them, which is now flattened.
### Changed:
- `graphics.overlay_boxes` now also accepts the `Axes` of a plain `sns.heatmap()` (its first argument is renamed from `clustermap_obj` to `target`). Cells hidden by seaborn (`mask=` or NaN) no longer get a box, and the default legend is placed past a colorbar on the right instead of over it.
- `graphics.overlay_boxes` (and `graphics.heatmap`'s `box_kws`) accept a single value for `sizes`, `facecolors`, `edgecolors` and `linewidths`, which is applied to every cell. Wrongly shaped inputs now raise a `ValueError` naming the argument. Sizes above 1 are allowed.
- `graphics.adjust_brightness` now always returns a hex string (e.g., `'#800000'`) instead of an RGB tuple, and has an `include_alpha` option (default `False`) to keep the alpha of the input color. It also accepts lists/arrays as input.
- `RSpace` keeps a 1x1 matrix as a 1x1 `pd.DataFrame` instead of collapsing it into a scalar, and integer matrices are now `int64` (were `int32`).
- `LinearModel.set_data` now sends categorical columns to R as factors, preserving their declared order and dropping unused categories. `remove_categories` and `preserve_na` arguments are removed. Ordered categoricals are kept ordered (R then uses polynomial contrasts) and a warning is logged.
- `LinearModel.set_reference` now raises a clear error for ordered factors.
- Logging: modules use `logging.getLogger(__name__)`, and only the `aa_utilities` root logger gets a (colored) handler. `loggers.setup_logger` is now idempotent, accepts `color` and `propagate`, and no longer takes `force`. `colorama` is no longer needed; set `NO_COLOR` to disable colors.

## [0.0.52] - 2026-10-01
### Added:
- `computation.modeling.LinearModel` now has a `get_contrast` that allows collecting relevant contrasts after computing them via `add_contrasts`.
- `storage.Checkpoint.latest` class method to easily resume the most recently created checkpoint folder.

## [0.0.51] - 2026-09-24
### Added:
- `aa_utilities.graphics.links` now produces the whole set of links in one go, allowing better spacing between the links
### Updated:
- `computation.modeling.LinearModel.add_contrasts` now supports `append` argument to separate compared levels in each contrast row.
- `RSpace` now captures R warnings across multiple calls via `capture_warnings()`
- `computation.modeling` section with improved documentation and added guards.

## [0.0.50] - 2026-09-03
### Added:
- Added `sankey` helper to create data needed for Sankey diagrams in plotly.

## [0.0.49] - 2026-06-29
### Fixed:
- `RSpace` now properly converts length(1) R objects to Pandas series.

## [0.0.48] - 2026-06-10
### Added:
- Added tests to prove that concurrency is possible in `RSpace`.
- `RSpace` now converts R objects to Python objects recursively, so nested R objects are now properly converted to Python objects.
- improved data conversion routes in `RSpace` to make it more efficient and robust.
- `RSpace` now has unified routes for data conversion. e.g., atomic vectors are now always converted to `pd.Series`, and lists are converted to `dict` or `list` depending on whether they are named or not.
- Added `watermark` which generates and returns a watermark string with system and environment info, similar to `session_info()` in R or the `watermark` package in Python.

## [0.0.47] - 2026-05-10
### Added:
- `computation.pairwise_associations` to compute pairwise sample associations depending on the type of pairs being compared.
- `RSpace` now returns `None` for R NULL values, irrespective of calling `R[]` or `R()`.

## [0.0.46] - 2026-04-22
### Added:
- `computation.pairwise_distances` to compute pairwise sample distances via multi-core implementation.
- `helpers.pandas.describe` to quickly visualize the content of a dataframe

## [0.0.45] - 2026-04-15
### Added:
- `helper.pandas.reorder_by_similarity` to reorder rows/columns by row/column similarity
- `computation.modeling.stability_selection` is now updated to accept any type of output.
- `graphics.overlay_boxes` to overlay boxes over existing `sns.clustermap` objects

## [0.0.44] - 2026-04-13
### Added:
- `graphics.UpsetPlot` to plot upset plots.

## [0.0.43] - 2026-03-31
### Added:
- `graphics.repel_text` to remove the overlap between textboxes.

## [0.0.42] - 2026-02-06
### Changed:
- `formatters` is now imported from `helpers`.
- `RSpace` now supports returning variables from nested objects in R and automatically convert them to Python objects
- `RSpace` now holds a copy of its dedicated `logger` in `self.logger`.
### Added:
- `RSpace.r_warnings` to collect the raised warnings.
- `graphics.to_colors` to convert a list of values to (hex) colors.
### Fixed:
- `graphics.heatmap` now uses the handlers currectly.
- `helpers.pandas.sort` now is much more readable, and handles `orders` properly.
- `loggers.RestrictedLogger` handler is now properly set. 

## [0.0.41] - 2026-01-23
### Added:
- `.computation.modeling.stability_selection` to check stability of selected features in a 
fitted model.

## [0.0.40] - 2025-11-26
### Changed:
- Moved all `pandas` related functions into a specific `.helpers.pandas` submodule.
### Added:
- `.helpers.pandas.search` function to search within a `pandas` dataframe.

## [0.0.39] - 2025-10-30
### Added:
- Added `Modeling.fit_negbin` to model response that follows negative binomial distribution.
- Added `RSpace` tests to confirm the validity of the conversion on both directions.

## [0.0.38] - 2025-10-08
### Changed:
- Changed the package name from `my_utilities` to `aa_utilities`

## [0.0.37] - 2025-10-22
### Added:
- `remove_effects`: To remove batch effects from a given gene expression vector.

## [0.0.36] - 2025-10-15
### Added:
- `storage.Checkpoint` to conveniently store Python objects in a time-annotated folder.
- `graphics.adjust_brightness` to lighten/darken a given color according to a given ratio.
- `Container` is now located in `storage` to make its location more intuitive.
- `jitter` to get (x-tick) offsets based on the neighborhood size.

## [0.0.35] - 2025-10-08
### Changed:
- `rpy2` implementation is now updated to remove bugs, enhance functionality, and improve its performance.

## [0.0.34] - 2025-10-02
### Added:
- `loggers.setup_logger()` to conveniently produce a logger with proper formatting and coloring.

### Changed:
- An overhaul of the `configurations` setup.
- `rpy2` is no longer in core require packages. Install with `

### Deprecited:
- `Container._params` is now depricated. Use `Container._pp.set_params` for setting printing
properties.

## [0.0.33] - 2025-09-19
### Changed:
- `TextWrapp` is now called `TextWrapper`.

## [0.0.32] - 2025-09-10
### Changed:
- Update on the codes to avoid importing many redundant dependencies until needed.

## [0.0.31] - 2025-06-04
### Added:
- `convenience.sort_by` to sort `pd.Series` or `pd.DataFrame` according to a given order.
### Changed:
- A minor update to `link` function to better adjust the `y-axis` after drawing the link, if needed.

## [0.0.30] - 2025-05-22
### Added:
- `stats.IterativeStats` to calculate `mean`, `var` and `std` of a list of numbers that 
are provided iteratively.
- `convenience.TextWrap` which is similar to the standard `textwrap` library, but it also 
allows preserving `newline` charachters.

### Changed:
- `is_true` can also receive a callable for `message` argument.

## [0.0.29] - 2025-04-01
### Changed:
- `Container` now represent its content in a `tree` like output

## [0.0.28] - 2025-03-26
### Added:
- `Container.update` adds/updates all given arguments as new elements in the container
- `filter`, `drop`, `copy`, `to_series` and `from_series` are also added.

### Changed:
- `Container` signature is now simplifed. It noq only receives elements using its constructor.
- `Container` now inherits from `dict` to simplify its operations
- `RSpace` now can transfer `dict` objects

## [0.0.27] - 2025-03-19
### Added:
- `text_offset` to add a text using an offset.

## [0.0.26] - 2025-03-07
### Fixed:
- `Logger` bug where `stdout` logs where counted as `info`.
- P-values in `forest_plot` is now represented with a single digit precision.
### Added:
- `convenience.Container` to hold data and represent them in a human-friendly manner.

## [0.0.25] - 2025-02-24
### Changed:
- `forest_plot` now has a `x_scale` argument to determine the scale of x-axis. By default the scale will be `linear`.

## [0.0.24] - 2025-02-19
### Added:
- `graphics.add_counts_to_legend` to add counts to the legend of a given ax.
- `graphics.heatmap` to add heatmaps that are further annotated with smaller colored boxes

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