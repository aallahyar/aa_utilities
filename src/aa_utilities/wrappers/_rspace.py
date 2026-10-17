import logging
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from rpy2 import (
    robjects as ro,
    rinterface as ri,
)
from rpy2.robjects import (
    numpy2ri,
    pandas2ri,
)

logger = logging.getLogger(__name__)

# Name of the hidden R closure (defined once per instance by `RSpace.__init__`) that runs a
# snippet and reports its outcome plus every warning/message it raised, see `RSpace.__call__`.
# `withCallingHandlers` + `invokeRestart` record each condition's full text (no console
# truncation/line-wrapping) and muffle it so R's own default handler never re-prints it; the
# `tryCatch` around it means an R error never escapes as a raw rpy2 exception, it's reported
# back as structured data (`status`/`error`) instead, alongside any warnings/messages raised
# before it. This changes nothing about the computation itself: `eval(parse(text=.code), ...)`
# runs in `globalenv()` exactly like `ro.r(code)` already does, and muffling a warning only
# skips R's console print of it (warnings/messages never affect a function's return value).
# `withVisible()` replicates `ro.r()`'s own `invisible=True` default (e.g. `x <- 1` reports a
# NULL value here, same as plain `ro.r("x <- 1")` returns `None`) -- without it, wrapping the
# result in `list(value = ...)` would make every result "visible" and change that behavior.
_RUNNER_NAME = '.aa_utilities_run'
_RUNNER_SOURCE = r"""
.aa_utilities_run <- function(.code) {
    .collector <- new.env(parent = emptyenv())
    .collector$warnings <- list()
    .collector$messages <- list()

    .record <- function(.cond, .bucket) {
        .call <- conditionCall(.cond)
        .entry <- list(
            message = conditionMessage(.cond),
            call = if (is.null(.call)) NULL else paste(deparse(.call), collapse = " "),
            classes = class(.cond)
        )
        .collector[[.bucket]] <- c(.collector[[.bucket]], list(.entry))
    }

    .outcome <- tryCatch(
        withCallingHandlers(
            {
                .visible <- withVisible(eval(parse(text = .code), envir = globalenv()))
                list(status = "ok", value = if (.visible$visible) .visible$value else NULL, error = NULL)
            },
            warning = function(w) { .record(w, "warnings"); invokeRestart("muffleWarning") },
            message = function(m) { .record(m, "messages"); invokeRestart("muffleMessage") }
        ),
        error = function(e) {
            .call <- conditionCall(e)
            list(
                status = "error",
                value = NULL,
                error = list(
                    message = conditionMessage(e),
                    call = if (is.null(.call)) NULL else paste(deparse(.call), collapse = " "),
                    classes = class(e)
                )
            )
        }
    )
    .outcome$warnings <- .collector$warnings
    .outcome$messages <- .collector$messages
    .outcome
}
"""


def _as_classes(value) -> tuple:
    """Normalizes a converted R `class(...)` vector (str, list, or pd.Series) into a tuple of str."""
    if value is None:
        return ()
    if isinstance(value, (list, tuple, pd.Series, np.ndarray)):
        return tuple(value)
    return (value,)


@dataclass(frozen=True)
class _RCondition:
    """Shared shape for a single structured R condition, with the snippet that raised it."""

    timestamp: datetime
    snippet: str
    message: str
    call: str | None = None
    classes: tuple[str, ...] = ()

    def __repr__(self):
        snippet = ' '.join(self.snippet.split())[:20]
        return f"[{self.timestamp:%H:%M:%S}] {snippet} | {self.message.strip()} "


class RWarning(_RCondition):
    """A single warning raised by R, with the snippet that triggered it."""


class RMessage(_RCondition):
    """A single `message()` emitted by R, with the snippet that triggered it."""


class RError(RuntimeError):
    """An error raised by R, carrying its structured condition (message/call/classes)."""

    def __init__(self, message: str, snippet: str = '', call: str | None = None, classes: tuple[str, ...] = ()):
        super().__init__(message)
        self.timestamp = datetime.now()
        self.snippet = snippet
        self.message = message
        self.call = call
        self.classes = classes

    def __repr__(self):
        snippet = ' '.join(self.snippet.split())[:20]
        return f"[{self.timestamp:%H:%M:%S}] {snippet} | {self.message.strip()} "


class RSpace:
    """A wrapper around `rpy2` package to facilitate import/export of variables between R and Python as well as running R commands.
    Most likely needed R packages are: install.packages(c('tidyverse', 'mmrm', 'MASS', 'emmeans'))

    Returns:
        RSpace object: An instance of the RSpace wrapper

    Examples:

    """

    _converter = ro.default_converter + numpy2ri.converter + pandas2ri.converter
    _ATOMIC_RTYPES = frozenset({
        ri.RTYPES.LGLSXP,
        ri.RTYPES.INTSXP,
        ri.RTYPES.REALSXP,
        ri.RTYPES.CPLXSXP,
        ri.RTYPES.STRSXP,
        ri.RTYPES.RAWSXP,
    })
    _r_names = ri.globalenv.find('names')
    _r_dim = ri.globalenv.find('dim')
    _r_rownames = ri.globalenv.find('rownames')
    _r_colnames = ri.globalenv.find('colnames')
    _r_is_na = ri.globalenv.find('is.na')

    def __init__(self, ipython=False):
        """Initiates an `R` environment.

        Args:
            ipython (bool, optional): It enables the `%R` magics command. See
                https://rpy2.github.io/doc/latest/html/interactive.html for details
        """
        self.ro = ro
        self.logger = logger
        self.ipython_loaded = ipython
        self.warnings: list[RWarning] = []  # append-only log of captured R warnings, see `capture_warnings()`
        self.messages: list[RMessage] = []  # append-only log of captured R messages, see `capture_messages()`

        # defines (or harmlessly redefines) the hidden runner closure used by `__call__`
        ro.r(_RUNNER_SOURCE)
        self._runner = ro.globalenv[_RUNNER_NAME]

        # loads IPython extension: https://rpy2.github.io/doc/latest/html/interactive.html#usage
        if ipython:
            # source: https://stackoverflow.com/questions/10361206/how-to-run-ipython-magic-from-a-script
            from IPython import get_ipython

            self.ipython_shell = get_ipython()
            if self.ipython_shell is not None:
                self.ipython_shell.run_line_magic('load_ext', 'rpy2.ipython')
                self.ipython_loaded = True

    def close(self):
        if self.ipython_loaded:
            self.ipython_shell.run_line_magic('unload_ext', 'rpy2.ipython')
            self.ipython_loaded = False

    def _py_to_r(self, value):
        """Recursively convert a Python value to an rpy2 object."""
        if isinstance(value, dict):
            return ro.r['list'](**{k: self._py_to_r(v) for k, v in value.items()})
        if isinstance(value, list):
            return ro.r['list'](*[self._py_to_r(v) for v in value])
        with self._converter.context():
            return ro.conversion.get_conversion().py2rpy(value)

    def __setitem__(self, name, value):
        ro.r.assign(name, self._py_to_r(value))

    @staticmethod
    def _to_python_atom(value):
        if hasattr(value, 'item'):
            try:
                return value.item()
            except Exception:
                pass
        return value

    def _flat_values(self, r_vec):
        """1-D array with the elements of an atomic R vector, or the column-major data of a matrix/array.

        Integer/logical vectors with NA become pandas nullable arrays, because rpy2 would otherwise
        leak the sentinel -2147483648 (integer) or a special NA object (logical).
        """
        typeof = r_vec.typeof
        is_na = np.array(list(self._r_is_na(r_vec)), dtype=bool)
        if typeof in (ri.RTYPES.INTSXP, ri.RTYPES.LGLSXP):
            is_int = typeof == ri.RTYPES.INTSXP
            values = [None if na else value for value, na in zip(r_vec, is_na)]
            if is_na.any():
                return pd.array(values, dtype='Int64' if is_int else 'boolean')
            return np.array(values, dtype=np.int64 if is_int else bool)
        if typeof == ri.RTYPES.STRSXP:
            return np.array([None if na else value for value, na in zip(r_vec, is_na)], dtype=object)
        dtype = {ri.RTYPES.REALSXP: float, ri.RTYPES.CPLXSXP: complex}.get(typeof, object)
        return np.array(list(r_vec), dtype=dtype)

    def _nullable_from_r(self, r_vec):
        """Returns a pandas nullable array for a plain R integer/logical vector that contains NA, else `None`."""
        if tuple(r_vec.rclass) not in {('integer',), ('logical',)}:
            return None
        values = self._flat_values(r_vec)
        return values if isinstance(values, pd.api.extensions.ExtensionArray) else None

    def _array_to_py(self, r_arr, dim):
        """Convert an atomic R matrix/array: 1-D → Series, 2-D → DataFrame, higher → numpy array."""
        flat = self._flat_values(r_arr)

        if len(dim) == 1:
            names = self._r_rownames(r_arr)
            return pd.Series(flat, index=None if names == ro.rinterface.NULL else list(names))

        if len(dim) == 2:
            n_row, n_col = dim
            value_py = pd.DataFrame({j: flat[j * n_row : (j + 1) * n_row] for j in range(n_col)})
            if self._r_rownames(r_arr) != ro.rinterface.NULL:
                value_py.index = list(self._r_rownames(r_arr))
            if self._r_colnames(r_arr) != ro.rinterface.NULL:
                value_py.columns = list(self._r_colnames(r_arr))
            return value_py

        if isinstance(flat, pd.api.extensions.ExtensionArray):
            flat = flat.to_numpy(dtype=object, na_value=pd.NA)
        return flat.reshape(dim, order='F')  # R stores arrays column-major

    def _coerce_atomic_scalar(self, value):
        if isinstance(value, ro.vectors.Vector):  # rpy2 leaves logical and complex vectors unconverted
            return self._to_python_atom(value[0])

        if isinstance(value, np.ndarray):
            if value.size == 1:
                return self._to_python_atom(value.reshape(-1)[0])
            return value

        # likely never needed,
        # if isinstance(value, pd.Series):
        #     if len(value) == 1:
        #         return self._to_python_atom(value.iloc[0])
        #     return value

        if isinstance(value, (list, tuple)):
            if len(value) == 1:
                return self._to_python_atom(value[0])
            return value

        return self._to_python_atom(value)

    def _r_to_py(self, r_obj):
        """Convert a raw rpy2 rinterface object to a Python value.
        Returns `None` for R NULL."""
        if r_obj is None or r_obj == ro.rinterface.NULL:
            return None

        typeof = r_obj.typeof
        is_named = self._r_names(r_obj) != ro.rinterface.NULL
        is_atomic = typeof in self._ATOMIC_RTYPES

        # Length-1 atomic scalar → Python scalar, but only when unnamed and without `dim`.
        # Named length-1 vectors (e.g. c(CSE = "0.0")) must fall through to the
        # pd.Series path below so the name is preserved; a 1x1 matrix stays a matrix.
        # https://stackoverflow.com/questions/38088392/how-do-you-check-for-a-scalar-in-r
        has_dim = self._r_dim(r_obj) != ro.rinterface.NULL
        if is_atomic and len(r_obj) == 1 and not is_named and not has_dim:
            if self._nullable_from_r(r_obj) is not None:  # a lone NA in an integer/logical vector
                return pd.NA
            with self._converter.context():
                value_rpy = ro.conversion.get_conversion().rpy2py(r_obj)
            return self._coerce_atomic_scalar(value_rpy)

        # VECSXP covers both data.frames and generic lists in R.
        # We branch on the R class attribute *before* calling rpy2py, because the pandas
        # converter would silently flatten a generic list into a pd.Series — losing structure.
        if typeof == ri.RTYPES.VECSXP:

            # case: a data.frame
            if 'data.frame' in list(r_obj.rclass):
                # data.frame → pd.DataFrame via the pandas converter
                with self._converter.context():
                    df = ro.conversion.get_conversion().rpy2py(r_obj)
                for position in range(len(r_obj)):
                    nullable = self._nullable_from_r(r_obj[position])
                    if nullable is not None:
                        df.isetitem(position, pd.Series(nullable, index=df.index))
                return df
            
            # case: a generic list (named: dict or unnamed: list), recursively convert elements
            # heterogeneous list: dict/list
            # homogeneous (atomic) vector: pd.Series
            names = self._r_names(r_obj)
            elements = [self._r_to_py(r_obj[i]) for i in range(len(r_obj))]
            if names != ro.rinterface.NULL:
                name_list = list(names)
                if len(name_list) != len(set(name_list)):
                    duplicates = {n for n in name_list if name_list.count(n) > 1}
                    raise ValueError(
                        f'R list has duplicate names {duplicates!r}, which cannot be represented as a Python dict.'
                    )
                return dict(zip(name_list, elements))
            return elements

        # Atomic matrices/arrays: rpy2 only reshapes integer/double ones, so they are rebuilt from their flat data
        if is_atomic and has_dim:
            return self._array_to_py(r_obj, [int(d) for d in self._r_dim(r_obj)])

        # All remaining R types: apply the converter
        with self._converter.context():
            value_rpy = ro.conversion.get_conversion().rpy2py(r_obj)

        # R atomic vectors → pd.Series (named index when names are present, RangeIndex otherwise).
        # Atomic vectors are homogeneous typed arrays with optional names, making pd.Series the natural Python analogue.
        # source: https://stackoverflow.com/questions/12944250/handing-null-return-in-rpy2
        # source: https://stackoverflow.com/questions/73259425/how-to-load-a-rtypes-nilsxp-data-object-when-using-rpy2
        if not has_dim:
            names = self._r_names(r_obj)
            data_vec = self._nullable_from_r(r_obj)
            if data_vec is None:
                data_vec = [self._to_python_atom(v) for v in value_rpy]
            if names == ro.rinterface.NULL:
                return pd.Series(data=data_vec, index=range(len(data_vec)))
            return pd.Series(data=data_vec, index=list(names))

        # non-atomic objects carrying `dim` (not expected here): fall back to rpy2's own conversion
        return value_rpy

    def __getitem__(self, name):
        try:
            r_obj = ri.globalenv.find(name)  # returns an rinterface-level object, no conversion yet
        except Exception as exc:
            raise KeyError(name) from exc
        return self._r_to_py(r_obj)

    def clear_warnings(self):
        """Discard all warnings captured so far."""
        self.warnings = []

    def clear_messages(self):
        """Discard all messages captured so far."""
        self.messages = []

    @contextmanager
    def capture_warnings(self):
        """Collect the `RWarning`s raised by any R() calls made inside this block.

        `self.warnings` is never reset automatically (see __call__), so this just
        watermarks its current length and slices off everything appended since.
        Usage:
            with self.capture_warnings() as warnings:
                self(...)
                self(...)
            # `warnings` now holds every RWarning raised across both calls
        """
        n_past_warnings = len(self.warnings)
        captured = []
        try:
            yield captured
        finally:
            captured.extend(self.warnings[n_past_warnings:])

    @contextmanager
    def capture_messages(self):
        """Collect the `RMessage`s raised by any R() calls made inside this block. Mirrors `capture_warnings()`."""
        n_past_messages = len(self.messages)
        captured = []
        try:
            yield captured
        finally:
            captured.extend(self.messages[n_past_messages:])

    def __call__(self, r_snippet, convert=True):
        """Runs `r_snippet` in R (in `globalenv()`, exactly like plain `ro.r(code)`) and returns its value.

        Every warning and `message()` call raised while running it is captured as a structured
        `RWarning`/`RMessage` (full, untruncated message text; one entry per condition, no matter
        how many lines its message spans) and appended to `self.warnings`/`self.messages` -
        including repeats, and including ones raised before an eventual error. An R error is
        raised here as an `RError`, carrying the same structured `message`/`call`/`classes` fields.
        """
        try:
            wrapped = self._runner(ro.StrVector([r_snippet]))
        except Exception as e:
            raise RuntimeError(f'R execution failed unexpectedly: {e}') from e

        fields = {name: position for position, name in enumerate(self._r_names(wrapped))}
        timestamp = datetime.now()

        new_warnings = self._r_to_py(wrapped[fields['warnings']]) or []
        self.warnings.extend(
            RWarning(
                timestamp=timestamp,
                snippet=r_snippet,
                message=entry['message'],
                call=entry.get('call'),
                classes=_as_classes(entry.get('classes')),
            )
            for entry in new_warnings
        )
        if new_warnings:
            self.logger.warning(
                '%d warning(s) issued during execution. Check `self.warnings` for details.', len(new_warnings)
            )

        new_messages = self._r_to_py(wrapped[fields['messages']]) or []
        self.messages.extend(
            RMessage(
                timestamp=timestamp,
                snippet=r_snippet,
                message=entry['message'].rstrip('\n'),  # message() always appends a trailing newline
                call=entry.get('call'),
                classes=_as_classes(entry.get('classes')),
            )
            for entry in new_messages
        )

        if self._r_to_py(wrapped[fields['status']]) == 'error':
            error = self._r_to_py(wrapped[fields['error']])
            raise RError(
                message=error['message'],
                snippet=r_snippet,
                call=error.get('call'),
                classes=_as_classes(error.get('classes')),
            )

        value = wrapped[fields['value']]
        if convert:
            try:
                return self._r_to_py(value)
            except Exception as e:
                raise RuntimeError(f'Failed to convert the returned object to Python: {e}.') from e
        return value

    def __repr__(self):
        var_infos = []
        for name in list(ri.globalenv):
            # value = ro.globalenv[name]
            value = ri.globalenv.find(name)  # returns an rinterface-level object, no conversion yet

            # Try to determine shape/length, if possible
            try:
                if hasattr(value, 'dim') and value.dim is not None:
                    shape = tuple(value.dim)
                elif hasattr(value, 'ncol') and hasattr(value, 'nrow'):
                    shape = (value.nrow, value.ncol)
                elif hasattr(value, 'length'):
                    shape = (value.length,)
                else:
                    shape = (len(value),)
            except Exception:
                shape = None
            var_infos.append(
                {
                    'name': name,
                    'type': type(value).__name__,
                    'shape': shape,
                }
            )
        var_infos = pd.DataFrame(var_infos)
        return str(var_infos)

    @classmethod
    def obj2cat(cls, dataframe: pd.DataFrame):
        converted = dataframe.copy()
        for col in converted.columns:
            if converted[col].dtype == object:
                converted[col] = converted[col].astype('category')
        return converted

    @classmethod
    def as_lines(cls, strings: list):
        return '\n'.join(map(str, strings))
