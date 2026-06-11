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
from rpy2.rinterface_lib import callbacks as rpy2_callbacks

from ..loggers import setup_logger
from .._configurations import configs

# setting up logger
logger = setup_logger(name='RSpace', level=configs.log.level)


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

    def __init__(self, ipython=False):
        """Initiates an `R` environment.

        Args:
            ipython (bool, optional): It enables the `%R` magics command. See
                https://rpy2.github.io/doc/latest/html/interactive.html for details
        """
        self.ro = ro
        self.logger = logger
        self.ipython_loaded = ipython
        self.warnings = []  # List to store captured R warnings

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

    def _coerce_atomic_scalar(self, value):
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

        # Length-1 atomic scalar → Python scalar.
        # https://stackoverflow.com/questions/38088392/how-do-you-check-for-a-scalar-in-r
        if typeof in self._ATOMIC_RTYPES and len(r_obj) == 1:
            with self._converter.context():
                value_rpy = ro.conversion.get_conversion().rpy2py(r_obj)
            return self._coerce_atomic_scalar(value_rpy)

        # VECSXP covers both data.frames and generic lists in R.
        # We branch on the R class attribute *before* calling rpy2py, because the pandas
        # converter would silently flatten a generic list into a pd.Series — losing structure.
        if typeof == ri.RTYPES.VECSXP:
            if 'data.frame' in list(r_obj.rclass):
                # data.frame → pd.DataFrame via the pandas converter
                with self._converter.context():
                    return ro.conversion.get_conversion().rpy2py(r_obj)
            # Generic lists → dict (named) or list (unnamed), fully recursive.
            # Generic lists are heterogeneous/recursive by design, so dict/list are the natural
            # Python analogues; atomic vectors are homogeneous typed arrays, so they map to
            # pd.Series (see below).
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

        # All remaining R types: apply the converter
        with self._converter.context():
            value_rpy = ro.conversion.get_conversion().rpy2py(r_obj)

        # check if the variable is more than 2D
        if isinstance(value_rpy, (np.ndarray,)) and value_rpy.ndim > 2:
            return value_rpy

        # R atomic vectors → pd.Series (named index when names are present, RangeIndex otherwise).
        # Atomic vectors are homogeneous typed arrays with optional names, making pd.Series the natural Python analogue.
        # source: https://stackoverflow.com/questions/12944250/handing-null-return-in-rpy2
        # source: https://stackoverflow.com/questions/73259425/how-to-load-a-rtypes-nilsxp-data-object-when-using-rpy2
        if self._r_dim(r_obj) == ro.rinterface.NULL:
            names = self._r_names(r_obj)
            data_vec = [self._to_python_atom(v) for v in value_rpy]
            if names == ro.rinterface.NULL:
                return pd.Series(data=data_vec, index=range(len(data_vec)))
            return pd.Series(data=data_vec, index=list(names))

        # 2D matrix → pd.DataFrame
        value_py = pd.DataFrame(data=value_rpy)
        if self._r_rownames(r_obj) != ro.rinterface.NULL:
            value_py.index = list(self._r_rownames(r_obj))
        if self._r_colnames(r_obj) != ro.rinterface.NULL:
            value_py.columns = list(self._r_colnames(r_obj))
        return value_py

    def __getitem__(self, name):
        try:
            r_obj = ri.globalenv.find(name)  # returns an rinterface-level object, no conversion yet
        except Exception as exc:
            raise KeyError(name) from exc
        return self._r_to_py(r_obj)

    def __call__(self, r_snippet, convert=True):
        self.warnings = []  # Reset warnings before execution

        # run the R script and capture warnings
        previous_warn_handler = rpy2_callbacks.consolewrite_warnerror
        rpy2_callbacks.consolewrite_warnerror = self.r_warn_handler  # Override the default warning/error writer
        try:
            returned_object = ro.r(r_snippet)
        except Exception as e:
            raise RuntimeError(f'R execution failed: {e}')
        finally:  # Ensure that we restore the original warning handler even if an error occurs
            rpy2_callbacks.consolewrite_warnerror = (
                previous_warn_handler  # Restore the original warning handler after execution
            )

        if len(self.warnings) != 0:  # If there were any warnings, log them
            self.logger.warning('Warning(s) issued during execution. Check `self.warnings` for details.')

        if convert:
            try:
                return self._r_to_py(returned_object)
            except Exception as e:
                raise RuntimeError(f'Failed to convert the returned object to Python: {e}.') from e
        return returned_object

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

    # capturing R warnings in a Python list
    def r_warn_handler(self, warning):
        self.warnings.append(warning)
        self.logger.warning(f'{warning}')  # Optional: still print it to console
