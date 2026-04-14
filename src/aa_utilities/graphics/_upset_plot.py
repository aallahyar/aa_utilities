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
    sets : dict[str, set]
        Ordered dictionary of set names to sets of elements.
        The key order determines the display order of sets.
    filter : callable, optional
        A function ``f(df) -> df`` that receives the intersections DataFrame
        and returns a filtered/modified copy.  Only the set columns (with
        values 1, 0, -1) need to be preserved; metadata columns (``_size``,
        ``_n_included``, ``_n_excluded``) are regenerated automatically.

        Common operations in the filter function:

        - Drop rows to remove intersections.
        - Drop columns to remove sets from the analysis.
        - Change cell values (e.g. ``0`` → ``-1``) to alter inclusion logic.
        - Reorder rows / columns to control display order.
    recompute : bool, default True
        Recompute ``_size`` from the original sets after filtering.  When
        ``False``, the ``_size`` values from the filter output are preserved
        (first occurrence kept on deduplication).
    mode : {'inclusive', 'exclusive'}, default 'inclusive'
        Default value for non-member sets in the generated matrix.

        - ``'inclusive'``: non-member sets are ``0`` (don't care).
        - ``'exclusive'``: non-member sets are ``-1`` (explicitly excluded).

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
        ``'bar_label'``, and ``'connector'`` (lines joining member dots).
        Useful for post-hoc customization via pandas slicing.
    """

    # ------------------------------------------------------------------ init

    def __init__(self, sets, filter=None, recompute=True, mode='inclusive'):
        if mode not in ('inclusive', 'exclusive'):
            raise ValueError(
                f"mode must be 'inclusive' or 'exclusive', got {mode!r}"
            )

        self._original_sets = {name: set(s) for name, s in sets.items()}
        self._set_names = list(sets.keys())

        self.sets = pd.DataFrame(
            {'size': {name: len(s) for name, s in self._original_sets.items()}}
        )

        self.intersections = self._generate_intersections(mode)

        if filter is not None:
            self.intersections = self._apply_filter(filter, recompute)

        # Populated by plot()
        self.fig = None
        self.ax_set_sizes = None
        self.ax_intersection_sizes = None
        self.ax_matrix = None
        self.artists = None

    # ------------------------------------------------------------------ data

    def _generate_intersections(self, mode):
        """Build the full intersection matrix (2^n - 1 rows)."""
        names = self._set_names
        non_member = -1 if mode == 'exclusive' else 0

        rows = []
        for r in range(1, len(names) + 1):
            for combo in combinations(names, r):
                included = set(combo)
                rows.append(
                    {name: (1 if name in included else non_member)
                     for name in names}
                )

        df = pd.DataFrame(rows)
        set_cols = list(names)
        df['_size'] = df.apply(
            lambda row: self._compute_size(row, set_cols), axis=1
        )
        df = self._refresh_meta(df, set_cols)
        df.index = self._make_labels(df, set_cols)
        df.index.name = None
        return df

    def _compute_size(self, row, set_cols):
        """Compute the cardinality of one intersection row."""
        included = [c for c in set_cols if row[c] == 1
                    and c in self._original_sets]
        excluded = [c for c in set_cols if row[c] == -1
                    and c in self._original_sets]

        if not included:
            return 0

        result = set.intersection(
            *(self._original_sets[c] for c in included)
        )
        if excluded:
            result = result - set.union(
                *(self._original_sets[c] for c in excluded)
            )
        return len(result)

    @staticmethod
    def _refresh_meta(df, set_cols):
        """(Re-)derive ``_n_included`` and ``_n_excluded``."""
        df = df.copy()  # avoid modifying original in-place
        df['_n_included'] = (df[set_cols] == 1).sum(axis=1)
        df['_n_excluded'] = (df[set_cols] == -1).sum(axis=1)
        return df

    @staticmethod
    def _make_labels(df, set_cols):
        """Derive row labels from the 1 / 0 / -1 pattern.

        Format: ``"A & B \\ C"``  (A intersect B, set-minus C).
        """
        labels = []
        for _, row in df.iterrows():
            included = [c for c in set_cols if row[c] == 1]
            excluded = [c for c in set_cols if row[c] == -1]

            parts = ' & '.join(included) if included else ''
            for ex in excluded:
                parts += f' \\ {ex}'
            labels.append(parts.strip() or '(empty)')
        return labels

    def _apply_filter(self, filter_fn, recompute):
        """Run the user's filter, relabel, deduplicate, refresh metadata."""
        df = filter_fn(self.intersections.copy())
        set_cols = [c for c in df.columns if c in self._original_sets]

        # Preserve _size before dropping meta (used when recompute=False)
        if not recompute and '_size' in df.columns:
            saved_sizes = df['_size'].values
            df = df[set_cols].copy()
            df['_size'] = saved_sizes
        else:
            df = df[set_cols].copy()

        # Relabel from surviving set columns
        df.index = self._make_labels(df, set_cols)

        # Deduplicate (keep first occurrence)
        df = df[~df.index.duplicated(keep='first')]

        # Sizes
        if recompute or '_size' not in df.columns:
            df['_size'] = df.apply(
                lambda row: self._compute_size(row, set_cols), axis=1
            )

        return self._refresh_meta(df, set_cols)

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
            Call :meth:`show` immediately.  Pass ``False`` to customise
            axes / artists before rendering.
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
        set_cols = [
            col for col in self.intersections.columns
            if col in self._original_sets
        ]
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
        self.fig, axes = plt.subplots(
            2, 2,
            figsize=figsize,
            gridspec_kw=dict(
                width_ratios=width_ratios,
                height_ratios=height_ratios,
                hspace=0.05,
                wspace=0.05,
            ),
        )
        axes[0, 0].axis('off')                          # spacer
        self.ax_intersection_sizes = axes[0, 1]
        self.ax_set_sizes = axes[1, 0]
        self.ax_matrix = axes[1, 1]

        # Share axes for alignment
        self.ax_matrix.sharex(self.ax_intersection_sizes)
        self.ax_matrix.sharey(self.ax_set_sizes)

        # ---- artist collection ----------------------------------------
        entries = []  # (set, intersection, element_type, artist)

        def _add(set_name, int_name, elem_type, artist):
            entries.append((set_name, int_name, elem_type, artist))

        # ---- intersection-size bars -----------------------------------
        self._plot_intersection_sizes(
            int_labels, sizes, n_ints, c, bar_labels, _add,
        )

        # ---- set-size bars --------------------------------------------
        self._plot_set_sizes(set_cols, n_sets, c, _add)

        # ---- matrix ---------------------------------------------------
        self._plot_matrix(
            set_cols, int_labels, n_sets, n_ints, c,
            marker_scale, figsize, row_stripe, col_stripe, _add,
        )

        # ---- build artists Series ------------------------------------
        idx = pd.MultiIndex.from_tuples(
            [(s, i, e) for s, i, e, _ in entries],
            names=['set', 'intersection', 'element'],
        )
        self.artists = pd.Series(
            [a for _, _, _, a in entries], index=idx,
        )

        self.fig.subplots_adjust(
            hspace=0.05, wspace=0.05,
            left=0.1, right=0.95, bottom=0.15, top=0.95,
        )

        if show:
            self.show()
        return self

    # ---- plot helpers (private) ----------------------------------------

    def _plot_intersection_sizes(
        self, int_labels, sizes, n_ints, c, bar_labels, _add,
    ):
        ax = self.ax_intersection_sizes
        xp = np.arange(n_ints)

        bars = ax.bar(xp, sizes, color=c['bar'], edgecolor='white', zorder=2)
        for i, bar in enumerate(bars):
            _add('', int_labels[i], 'bar', bar)

        if bar_labels:
            for i in range(n_ints):
                txt = ax.text(
                    xp[i], sizes[i], str(int(sizes[i])),
                    ha='center', va='bottom', fontsize=8,
                )
                _add('', int_labels[i], 'bar_label', txt)

        ax.set_ylabel('Intersection size')
        ax.tick_params(bottom=False, labelbottom=False)
        for spine in ('top', 'right', 'bottom'):
            ax.spines[spine].set_visible(False)

    def _plot_set_sizes(self, set_cols, n_sets, c, _add):
        ax = self.ax_set_sizes
        yp = np.arange(n_sets)
        ss = self.sets.loc[set_cols, 'size'].values

        bars = ax.barh(yp, ss, color=c['bar'], edgecolor='white', zorder=2)
        for i, bar in enumerate(bars):
            _add(set_cols[i], '', 'bar', bar)

        ax.set_yticks(yp)
        ax.set_yticklabels(set_cols)
        ax.invert_xaxis()
        ax.set_xlabel('Set size')
        for spine in ('top', 'right', 'left'):
            ax.spines[spine].set_visible(False)

    def _plot_matrix(
        self, set_cols, int_labels, n_sets, n_ints, c,
        marker_scale, figsize, row_stripe, col_stripe, _add,
    ):
        ax = self.ax_matrix
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
        dot_area = base ** 2

        color_map = {1: c['member'], -1: c['excluded'], 0: c['ignored']}

        # Stripes — use alpha so row and column stripes blend where they
        # overlap instead of one fully overwriting the other.
        if row_stripe:
            for i in range(0, n_sets, 2):
                ax.axhspan(
                    i - 0.5, i + 0.5,
                    facecolor=row_stripe, edgecolor='none',
                    alpha=0.5, zorder=0,
                )
                self.ax_set_sizes.axhspan(
                    i - 0.5, i + 0.5,
                    facecolor=row_stripe, edgecolor='none',
                    alpha=0.5, zorder=0,
                )
        if col_stripe:
            for i in range(0, n_ints, 2):
                ax.axvspan(
                    i - 0.5, i + 0.5,
                    facecolor=col_stripe, edgecolor='none',
                    alpha=0.5, zorder=0,
                )
                self.ax_intersection_sizes.axvspan(
                    i - 0.5, i + 0.5,
                    facecolor=col_stripe, edgecolor='none',
                    alpha=0.5, zorder=0,
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
                    ix, sx,
                    s=dot_area,
                    c=color_map.get(val, c['ignored']),
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
                    ix, sx,
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
            int_labels, rotation=90, ha='center', fontsize=8,
        )
        ax.set_xlim(-0.5, n_ints - 0.5)
        ax.set_ylim(n_sets - 0.5, -0.5)  # inverted y
        ax.tick_params(left=False, bottom=False, labelleft=False)
        for spine in ax.spines.values():
            spine.set_visible(False)

    # ------------------------------------------------------------------ show

    def show(self):
        """Display the figure via ``plt.show()``."""
        if self.fig is not None:
            plt.show()

