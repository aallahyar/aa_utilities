"""A Python implementation of ComplexUpset plot using matplotlib.

Reference: https://github.com/krassowski/complex-upset
"""

from itertools import combinations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class UpsetPlot:
    """UpSet plot for visualizing set intersections.

    Parameters
    ----------
    sets : dict[str, set | list | iterable]
        Ordered dictionary of set names to sets of elements.
        The key order determines the display order of sets.
    mode : {'inclusive', 'exclusive'}, default 'inclusive'
        Default value for non-member sets in the generated matrix.

        - ``'inclusive'``: non-member sets are ``0`` (don't care).
        - ``'exclusive'``: non-member sets are ``-1`` (explicitly excluded).
    min_included_sets : int or None, default None
        Minimum number of included sets (value ``1``) required for an
        intersection to be generated/kept. ``None`` defaults to ``1``.
    max_included_sets : int or None, default None
        Maximum number of included sets (value ``1``) allowed for an
        intersection to be generated/kept. ``None`` means no upper bound.
    min_intersection_size : int or None, default None
        Minimum overlap size (cardinality of the intersection) required
        for an intersection to be generated/kept. ``None`` means no
        lower bound.
    max_intersection_size : int or None, default None
        Maximum overlap size (cardinality of the intersection) allowed
        for an intersection to be generated/kept. ``None`` means no
        upper bound.

    Attributes
    ----------
    sets : pd.DataFrame
        One row per input set with a ``size`` column.
    intersections : pd.DataFrame
        One row per intersection.  Set columns contain ``1`` (member),
        ``0`` (ignored) or ``-1`` (excluded).  Metadata columns:
        ``_size``, ``_n_included``, ``_n_excluded``.
    fig : Figure or None
        Matplotlib Figure, available after :meth:`plot`.
    ax_set_sizes : Axes or None
        Axes for the horizontal set-size bars (left panel).
    ax_intersection_sizes : Axes or None
        Axes for the vertical intersection-size bars (top panel).
    ax_matrix : Axes or None
        Axes for the dot-matrix (bottom-right panel).  Each intersection
        column shows dots for every set, coloured by role (member,
        excluded, ignored).  Vertical *connectors* link the member dots
        within each intersection.
    artists : pd.Series or None
        Multi-indexed ``(set, intersection, element)`` Series of matplotlib
        artists.  ``element`` values are ``'dot'``, ``'bar'``,
        ``'intersection_size_label'`` (intersection-size bar labels, top panel),
        ``'set_size_label'`` (set-size bar labels, left panel), and
        ``'connector'`` (lines joining member dots).
        Useful for post-hoc customization via pandas slicing.
    """

    # ------------------------------------------------------------------ init

    def __init__(
        self,
        sets,
        mode='inclusive',
        min_included_sets=None,
        max_included_sets=None,
        min_intersection_size=None,
        max_intersection_size=None,
    ):
        if mode not in ('inclusive', 'exclusive'):
            raise ValueError(f"mode must be 'inclusive' or 'exclusive', got {mode!r}")

        self.sets = pd.DataFrame(
            {
                'members': [set(s) for s in sets.values()],
                'size': [len(s) for s in sets.values()],
            },
            index=sets.keys(),
        )

        self.min_included_sets, self.max_included_sets = self._validate_included_set_limits(
            min_included_sets,
            max_included_sets,
        )
        self.min_intersection_size, self.max_intersection_size = self._validate_intersection_size_limits(
            min_intersection_size,
            max_intersection_size,
        )

        self.intersections = self._generate_intersections(mode)

        # Populated by plot()
        self.figure = None
        self.ax_set_sizes = None
        self.ax_intersection_sizes = None
        self.ax_matrix = None
        self.artists = None

    # ------------------------------------------------------------------ data

    @staticmethod
    def _validate_included_set_limits(min_included_sets, max_included_sets):
        """Validate constructor arguments controlling included-set bounds."""
        if min_included_sets is None:
            min_included_sets = 1
        if max_included_sets is not None and not isinstance(max_included_sets, int):
            raise ValueError('max_included_sets must be an integer or None')
        if not isinstance(min_included_sets, int):
            raise ValueError('min_included_sets must be an integer or None')
        if min_included_sets < 1:
            raise ValueError('min_included_sets must be >= 1')
        if max_included_sets is not None and max_included_sets < 1:
            raise ValueError('max_included_sets must be >= 1 when provided')
        if max_included_sets is not None and min_included_sets > max_included_sets:
            raise ValueError('min_included_sets must be <= max_included_sets')
        return min_included_sets, max_included_sets

    def _effective_max_included_sets(self, n_set_cols):
        """Upper bound capped by available set columns."""
        if self.max_included_sets is None:
            return n_set_cols
        return min(self.max_included_sets, n_set_cols)

    @staticmethod
    def _validate_intersection_size_limits(min_intersection_size, max_intersection_size):
        """Validate constructor arguments controlling overlap-size bounds."""
        if min_intersection_size is not None and not isinstance(min_intersection_size, int):
            raise ValueError('min_intersection_size must be an integer or None')
        if max_intersection_size is not None and not isinstance(max_intersection_size, int):
            raise ValueError('max_intersection_size must be an integer or None')
        if min_intersection_size is not None and min_intersection_size < 0:
            raise ValueError('min_intersection_size must be >= 0 when provided')
        if max_intersection_size is not None and max_intersection_size < 0:
            raise ValueError('max_intersection_size must be >= 0 when provided')
        if (
            min_intersection_size is not None
            and max_intersection_size is not None
            and min_intersection_size > max_intersection_size
        ):
            raise ValueError('min_intersection_size must be <= max_intersection_size')
        return min_intersection_size, max_intersection_size

    def _is_intersection_size_allowed(self, size):
        """Check whether an overlap cardinality is inside configured bounds."""
        if self.min_intersection_size is not None and size < self.min_intersection_size:
            return False
        if self.max_intersection_size is not None and size > self.max_intersection_size:
            return False
        return True

    def _apply_intersection_size_limits(self, ixs):
        """Keep rows within configured overlap-size bounds."""
        if '_size' not in ixs.columns:
            return ixs

        keep = pd.Series(True, index=ixs.index)
        if self.min_intersection_size is not None:
            keep &= ixs['_size'] >= self.min_intersection_size
        if self.max_intersection_size is not None:
            keep &= ixs['_size'] <= self.max_intersection_size
        return ixs.loc[keep].copy()

    def _compute_size_from_parts(self, included, excluded):
        """Compute overlap cardinality from included/excluded set labels."""
        if not included:
            return 0

        result = set.intersection(*(self.sets.loc[list(included), 'members'].tolist()))
        if excluded:
            result = result - set.union(*(self.sets.loc[list(excluded), 'members'].tolist()))
        return len(result)

    def _apply_included_set_limits(self, ixs, set_cols):
        """Keep rows within configured included-set bounds."""
        if not set_cols:
            return ixs.iloc[0:0].copy()

        min_k = self.min_included_sets
        max_k = self._effective_max_included_sets(len(set_cols))
        n_included = (ixs[set_cols] == 1).sum(axis=1)
        keep = (n_included >= min_k) & (n_included <= max_k)
        return ixs.loc[keep].copy()

    def _generate_intersections(self, mode):
        """Build the full intersection matrix (2^n - 1 rows)."""
        set_labels = list(self.sets.index)
        non_member = -1 if mode == 'exclusive' else 0
        all_labels = set(set_labels)
        min_k = self.min_included_sets
        max_k = self._effective_max_included_sets(len(set_labels))

        rows = []
        for r in range(min_k, max_k + 1):
            for combo in combinations(set_labels, r):
                included = set(combo)
                excluded = all_labels - included if mode == 'exclusive' else set()
                size = self._compute_size_from_parts(included, excluded)
                if not self._is_intersection_size_allowed(size):
                    continue

                row = {name: (1 if name in included else non_member) for name in set_labels}
                row['_size'] = size
                rows.append(row)

        ixs = pd.DataFrame(rows, columns=[*set_labels, '_size'])
        ixs = self._refresh_meta(ixs, set_labels)
        ixs.index = self._derive_labels(ixs, set_labels)
        ixs.index.name = None
        return ixs

    def filter(self, fn, update_stats=True):
        """Apply a filter function to the intersections.

        Parameters
        ----------
        fn : callable
            A function ``f(ix) -> ix`` that receives a copy of
            :attr:`intersections` and returns a modified DataFrame.
            Only the set columns (with values 1, 0, -1) need to be
            preserved; metadata columns are regenerated automatically.

            Common operations::

                # keep large intersections, sorted by size
                up.filter(lambda ix: ix[ix['_size'] >= 5]
                                       .sort_values('_size', ascending=False))

                # drop a set column
                up.filter(lambda ix: ix.drop(columns=['Gene Set D']))

                # make all non-members exclusive
                up.filter(lambda ix: ix.replace(0, -1))

        update_stats : bool, default True
            Recompute ``_size`` from the original sets after filtering.
            When ``False``, the ``_size`` values from the filter output
            are preserved (first occurrence kept on deduplication).

        Returns
        -------
        self
            For method chaining.
        """
        ixs = fn(self.intersections.copy())
        set_cols = [c for c in ixs.columns if c in self.sets.index]

        # Preserve _size before dropping meta (used when update_stats=False)
        if not update_stats and '_size' in ixs.columns:
            saved_sizes = ixs['_size'].values
            ixs = ixs[set_cols].copy()
            ixs['_size'] = saved_sizes
        else:
            ixs = ixs[set_cols].copy()

        # Relabel from surviving set columns
        ixs.index = self._derive_labels(ixs, set_cols)

        # Deduplicate (keep first occurrence)
        # : improves readability by making it clear that we're dropping duplicate rows, not columns
        ixs = ixs.loc[~ixs.index.duplicated(keep='first'), :]

        # Respect included-set bounds configured at construction.
        ixs = self._apply_included_set_limits(ixs, set_cols)

        # Sizes
        if update_stats or '_size' not in ixs.columns:
            ixs['_size'] = ixs.apply(lambda row: self._compute_size(row, set_cols), axis=1)

        # Respect overlap-size bounds configured at construction.
        ixs = self._apply_intersection_size_limits(ixs)

        self.intersections = self._refresh_meta(ixs, set_cols)
        return self

    def _compute_size(self, row, set_cols):
        """Compute the cardinality of one intersection row."""
        included = [c for c in set_cols if row[c] == 1]
        excluded = [c for c in set_cols if row[c] == -1]

        if not included:
            return 0

        result = set.intersection(
            *(self.sets.loc[included, 'members'].tolist()),
        )
        if excluded:
            result = result - set.union(*(self.sets.loc[excluded, 'members'].tolist()))
        return len(result)

    @staticmethod
    def _refresh_meta(ixs, set_cols):
        """(Re-)derive ``_n_included`` and ``_n_excluded``."""
        ixs = ixs.copy()  # avoid modifying original in-place
        ixs['_n_included'] = (ixs[set_cols] == 1).sum(axis=1)
        ixs['_n_excluded'] = (ixs[set_cols] == -1).sum(axis=1)
        return ixs

    @staticmethod
    def _derive_labels(ixs, set_cols):
        """Derive row labels from the 1 / 0 / -1 pattern.

        Format: ``"A & B \\ C"``  (A intersect B, set-minus C).
        """
        labels = []
        for _, row in ixs.iterrows():
            included = [c for c in set_cols if row[c] == 1]
            excluded = [c for c in set_cols if row[c] == -1]

            if not included: # empty intersection
                labels.append('(empty)')
            else:
                parts = ' & '.join(included)
                for ex in excluded:
                    parts += f' \\ {ex}'
                labels.append(parts)
        return labels

    # ------------------------------------------------------------------ plot

    _DEFAULT_COLORS = {
        'member': '#000000',
        'excluded': '#cc3333',
        'ignored': '#e0e0e0',
        'connector': '#000000',
        'bar': '#3366cc',
    }

    def plot(
        self,
        *,
        show=True,
        row_stripe='#f5f5f5',
        col_stripe=None,
        bar_labels=True,
        marker_scale=1.0,
        colors=None,
        figsize=None,
        width_ratios=(0.25, 0.75),
        height_ratios=(0.6, 0.4),
    ):
        """Create the UpSet figure.

        Parameters
        ----------
        show : bool, default True
            Call :meth:`show` immediately.  Pass ``False`` to defer rendering which in turn allows 
            access and customization of axes / artists before rendering.
        row_stripe : str or None, default '#f5f5f5'
            Alternating row-stripe colour.  ``None`` disables.
        col_stripe : str or None, default None
            Alternating column-stripe colour.  ``None`` disables.
        bar_labels : bool, default True
            Show count labels above intersection-size bars.
        marker_scale : float, default 1.0
            Multiplier for dot size in the matrix.
        colors : dict, optional
            Override default colours.  Recognised keys: ``'member'``,
            ``'excluded'``, ``'ignored'``, ``'connector'``, ``'bar'``.
        figsize : tuple, optional
            ``(width, height)`` in inches.  Auto-computed when ``None``.
        width_ratios : tuple, default (0.25, 0.75)
            Relative widths of (set-size panel, matrix panel).
        height_ratios : tuple, default (0.6, 0.4)
            Relative heights of (intersection-size panel, matrix panel).

        Returns
        -------
        self
            For method chaining.
        """
        c = {**self._DEFAULT_COLORS, **(colors or {})}
        set_cols = [col for col in self.intersections.columns if col in self.sets.index]
        n_sets = len(set_cols)
        n_ints = len(self.intersections)
        int_labels = self.intersections.index.tolist()
        sizes = self.intersections['_size'].values

        if figsize is None:
            figsize = (
                max(8, n_ints * 0.6 + 2),
                max(5, n_sets * 0.5 + 3),
            )

        # ---- layout --------------------------------------------------
        self.figure, axes = plt.subplots(
            ncols=2,
            nrows=2,
            figsize=figsize,
            gridspec_kw=dict(
                width_ratios=width_ratios,
                height_ratios=height_ratios,
            ),
        )
        axes[0, 0].axis('off')  # spacer
        self.ax_intersection_sizes = axes[0, 1]
        self.ax_set_sizes = axes[1, 0]
        self.ax_matrix = axes[1, 1]

        # Share axes for alignment
        self.ax_matrix.sharex(self.ax_intersection_sizes)
        self.ax_matrix.sharey(self.ax_set_sizes)

        # ---- artist collection ----------------------------------------
        entries = []  # (set, intersection, element_type, artist)

        def _add(set_label, int_label, elem_type, artist):
            entries.append((set_label, int_label, elem_type, artist))

        # ---- intersection-size bars -----------------------------------
        self._plot_intersection_sizes(
            int_labels,
            sizes,
            c,
            bar_labels,
            _add,
        )

        # ---- set-size bars --------------------------------------------
        self._plot_set_sizes(set_cols, c, _add)

        # ---- matrix ---------------------------------------------------
        self._plot_matrix(
            set_cols,
            int_labels,
            c,
            marker_scale,
            figsize,
            row_stripe,
            col_stripe,
            _add,
        )

        # ---- build artists Series ------------------------------------
        idx = pd.MultiIndex.from_tuples(
            [(s, i, e) for s, i, e, _ in entries],
            names=['set', 'intersection', 'element'],
        )
        self.artists = pd.Series(
            [a for _, _, _, a in entries],
            index=idx,
        )

        self.figure.subplots_adjust(
            hspace=0.05,
            wspace=0.05,
            left=0.1,
            right=0.95,
            bottom=0.15,
            top=0.95,
        )

        if show:
            self.show()
        return self

    # ---- plot helpers (private) ----------------------------------------

    def _plot_intersection_sizes(
        self,
        int_labels,
        sizes,
        c,
        bar_labels,
        _add,
    ):
        ax = self.ax_intersection_sizes
        n_ints = len(sizes)
        xp = np.arange(n_ints)

        bars = ax.bar(xp, sizes, color=c['bar'], edgecolor='white', zorder=2)
        for i, bar in enumerate(bars):
            _add('', int_labels[i], 'bar', bar)

        if bar_labels:
            for i in range(n_ints):
                txt = ax.text(
                    xp[i],
                    sizes[i],
                    str(int(sizes[i])),
                    ha='center',
                    va='bottom',
                    fontsize=8,
                )
                _add('', int_labels[i], 'intersection_size_label', txt)

        ax.set_ylabel('Intersection size')
        ax.tick_params(bottom=False, labelbottom=False)
        for spine in ('top', 'right', 'bottom'):
            ax.spines[spine].set_visible(False)

    def _plot_set_sizes(self, set_cols, c, _add):
        ax = self.ax_set_sizes
        n_sets = len(set_cols)
        yp = np.arange(n_sets)
        ss = self.sets.loc[set_cols, 'size'].values

        bars = ax.barh(yp, ss, color=c['bar'], edgecolor='white', zorder=2)
        for i, bar in enumerate(bars):
            _add(set_cols[i], '', 'bar', bar)

        for i, (bar, size) in enumerate(zip(bars, ss)):
            txt = ax.text(
                bar.get_width(),
                yp[i],
                f'{int(size)} ',
                ha='right',
                va='center',
                fontsize=8,
                zorder=3,
            )
            _add(set_cols[i], '', 'set_size_label', txt)

        ax.set_yticks(yp)
        ax.set_yticklabels(set_cols)
        ax.set_ylim(n_sets - 0.5, -0.5)
        ax.invert_xaxis()
        ax.set_xlim(left=max(ss) * 1.15)  # padding so size labels don't overlap set names
        ax.set_xlabel('Set size')
        for spine in ('top', 'right', 'left'):
            ax.spines[spine].set_visible(False)

    def _plot_matrix(
        self,
        set_cols,
        int_labels,
        c,
        marker_scale,
        figsize,
        row_stripe,
        col_stripe,
        _add,
    ):
        ax = self.ax_matrix
        n_sets = len(set_cols)
        n_ints = len(int_labels)
        mat = self.intersections[set_cols].values  # (n_ints, n_sets)

        # Dot size adapts to grid density
        base = (
            min(
                figsize[0] * 72 / max(n_ints, 1),
                figsize[1] * 72 / max(n_sets, 1),
            )
            * 0.3
            * marker_scale
        )
        dot_area = base**2

        color_map = {-1: c['excluded'], 0: c['ignored']} # members (1) are handled separately

        # Stripes — use alpha so row and column stripes blend where they
        # overlap instead of one fully overwriting the other.
        if row_stripe:
            for i in range(0, n_sets, 2):
                ax.axhspan(
                    i - 0.5,
                    i + 0.5,
                    facecolor=row_stripe,
                    edgecolor='none',
                    alpha=0.5,
                    zorder=0,
                )
                self.ax_set_sizes.axhspan(
                    i - 0.5,
                    i + 0.5,
                    facecolor=row_stripe,
                    edgecolor='none',
                    alpha=0.5,
                    zorder=0,
                )
        if col_stripe:
            for i in range(0, n_ints, 2):
                ax.axvspan(
                    i - 0.5,
                    i + 0.5,
                    facecolor=col_stripe,
                    edgecolor='none',
                    alpha=0.5,
                    zorder=0,
                )
                self.ax_intersection_sizes.axvspan(
                    i - 0.5,
                    i + 0.5,
                    facecolor=col_stripe,
                    edgecolor='none',
                    alpha=0.5,
                    zorder=0,
                )

        # Dots and connectors — draw non-member dots first (lower zorder),
        # then connectors, then member dots on top.
        for ix in range(n_ints):
            member_ys = []
            for sx in range(n_sets):
                val = int(mat[ix, sx])
                if val == 1:
                    member_ys.append(sx)
                    continue  # draw member dots after the connector
                dot = ax.scatter(
                    ix,
                    sx,
                    s=dot_area,
                    c=color_map[val],
                    marker='o',
                    zorder=2,
                    edgecolors='none',
                )
                _add(set_cols[sx], int_labels[ix], 'dot', dot)

            # Connector on top of non-member dots
            if len(member_ys) >= 2:
                line = ax.plot(
                    [ix, ix],
                    [min(member_ys), max(member_ys)],
                    color=c['connector'],
                    linewidth=3,
                    zorder=3,
                    solid_capstyle='round',
                )[0]
                _add('', int_labels[ix], 'connector', line)

            # Member dots on top of connector
            for sx in member_ys:
                dot = ax.scatter(
                    ix,
                    sx,
                    s=dot_area,
                    c=c['member'],
                    marker='o',
                    zorder=4,
                    edgecolors='none',
                )
                _add(set_cols[sx], int_labels[ix], 'dot', dot)

        # Axis formatting
        xp = np.arange(n_ints)
        ax.set_xticks(xp)
        ax.set_xticklabels(
            int_labels,
            rotation=90,
            ha='center',
            fontsize=10,
        )
        ax.set_xlim(-0.5, n_ints - 0.5)
        ax.set_ylim(n_sets - 0.5, -0.5)  # inverted y
        ax.tick_params(left=False, bottom=False, labelleft=False)
        for spine in ax.spines.values():
            spine.set_visible(False)

    # ------------------------------------------------------------------ show

    def show(self):
        """Display the figure via ``plt.show()``."""
        if self.figure is not None:
            # if interactive backend, this will render all figures; 
            # in that case, use self.fig.show()
            plt.show()
