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
import rpy2.rlike.container as rlc
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

    def __setitem__(self, name, value):
        if isinstance(value, (dict,)):
            value = rlc.NamedList.from_items(value)
            ro.r.assign(name, value)
            self(f'''
            {name} <- lapply({name}, unlist, use.names=FALSE, recursive=FALSE)
            ''')  # Convert each value to regular list in R to avoid rpy2 issues with NamedList
        else:
            with (ro.default_converter + numpy2ri.converter + pandas2ri.converter).context():
                value_r = ro.conversion.get_conversion().py2rpy(value)
                ro.r.assign(name, value_r)

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
        if r_obj == ro.rinterface.NULL:
            return None

        # performing type conversions
        with (ro.default_converter + numpy2ri.converter + pandas2ri.converter).context():
            value_rpy = ro.conversion.get_conversion().rpy2py(r_obj)

        # get R functions for type checking and metadata retrieval
        r_is_atomic = ri.globalenv.find('is.atomic')
        r_length = ri.globalenv.find('length')
        r_dim = ri.globalenv.find('dim')
        r_names = ri.globalenv.find('names')
        r_rownames = ri.globalenv.find('rownames')
        r_colnames = ri.globalenv.find('colnames')

        # check if the variable is scalar: https://stackoverflow.com/questions/38088392/how-do-you-check-for-a-scalar-in-r
        # pure Python (with identically item types) scalar, but also lists/arrays (with length 1) would be caught here
        if r_is_atomic(r_obj)[0] and r_length(r_obj)[0] == 1:
            return self._coerce_atomic_scalar(value_rpy)

        # check if the variable is already typed properly
        if isinstance(value_rpy, (pd.DataFrame,)):
            return value_rpy

        # do we have an array of Strings? No longer needed as is covered in pd.Series part
        # if isinstance(value_rpy, ro.vectors.StrVector):
        #     return np.array(value_rpy)

        # check if the variable is more than 2D
        if isinstance(value_rpy, (np.ndarray,)) and value_rpy.ndim > 2:
            return value_rpy

        # adding column names if present
        # source: https://stackoverflow.com/questions/12944250/handing-null-return-in-rpy2
        # source: https://stackoverflow.com/questions/73259425/how-to-load-a-rtypes-nilsxp-data-object-when-using-rpy2
        if r_dim(r_obj) == ro.rinterface.NULL or np.array(value_rpy).ndim == 1:
            names = r_names(r_obj)
            data_vec = [self._to_python_atom(v) for v in value_rpy]
            if names == ro.rinterface.NULL:
                value_py = pd.Series(
                    data=data_vec,
                    index=range(len(data_vec)),
                )
            else:
                value_py = pd.Series(
                    data=data_vec,
                    index=list(names),
                )
        else:
            value_py = pd.DataFrame(
                data=value_rpy,
            )
            if r_rownames(r_obj) != ro.rinterface.NULL:
                value_py.index = list(r_rownames(r_obj))
            if r_colnames(r_obj) != ro.rinterface.NULL:
                value_py.columns = list(r_colnames(r_obj))

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
