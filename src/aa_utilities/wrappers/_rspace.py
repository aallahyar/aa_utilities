from os import name
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

from ..loggers import setup_logger
from .._configurations import configs

# setting up logger
logger = setup_logger(name=__name__, level=configs.log.level)

class RSpace():
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
        self.ipython_loaded = ipython
        
        # loads IPython extension: https://rpy2.github.io/doc/latest/html/interactive.html#usage
        if ipython:
            # source: https://stackoverflow.com/questions/10361206/how-to-run-ipython-magic-from-a-script
            from IPython import get_ipython
            self.ipython_shell = get_ipython()
            if self.ipython_shell is not None:
                self.ipython_shell.run_line_magic("load_ext", "rpy2.ipython")
                self.ipython_loaded = True

    def close(self):
        if self.ipython_loaded:
            self.ipython_shell.run_line_magic("unload_ext", "rpy2.ipython")
            self.ipython_loaded = False

    def __setitem__(self, name, value):
        if isinstance(value, (dict, )):
            value = rlc.NamedList.from_items(value)
            ro.r.assign(name, value)
        else:
            with (ro.default_converter + numpy2ri.converter + pandas2ri.converter).context():
                value_r = ro.conversion.get_conversion().py2rpy(value)
                ro.r.assign(name, value_r)

    def __getitem__(self, name):

        # fetch raw R object first (no conversion)
        r_obj = ri.globalenv.find(name) # returns an rinterface-level object, no conversion yet
        
        # performing type conversions
        with (ro.default_converter + numpy2ri.converter + pandas2ri.converter).context():
            value_rpy = ro.conversion.get_conversion().rpy2py(ro.globalenv[name])
        
        # check if the variable is scalar: https://stackoverflow.com/questions/38088392/how-do-you-check-for-a-scalar-in-r
        # pure Python (with identically item types) scalar, but also lists/arrays (with length 1) would be caught here
        if ri.globalenv.find('is.atomic')(r_obj)[0] and ri.globalenv.find('length')(r_obj)[0] == 1:
            # check in Python side as well to make sure it is a single-element list/array
            if len(value_rpy) == 1: # isinstance(value_rpy, (list, tuple, np.ndarray, )) and 
                # assert len(value_rpy) == 1, f'Unexpected length for a scalar variable: {value_rpy}'
                return value_rpy[0].item()
            else: # pure Python (with identically item types) list/array with length > 1
                # raise ValueError
                logger.warning(f'Unexpected length for a scalar variable: {value_rpy}')
                return np.array(value_rpy)

        # check if the variable is already typed properly
        if isinstance(value_rpy, (pd.DataFrame, )):
            return value_rpy
        
        # do we have an array of Strings? No longer needed as is covered in pd.Series part
        # if isinstance(value_rpy, ro.vectors.StrVector):
        #     return np.array(value_rpy)
        
        # check if the variable is more than 2D
        if isinstance(value_rpy, (np.ndarray, )) and value_rpy.ndim > 2:
            return value_rpy

        # adding column names if present
        # source: https://stackoverflow.com/questions/12944250/handing-null-return-in-rpy2
        # source: https://stackoverflow.com/questions/73259425/how-to-load-a-rtypes-nilsxp-data-object-when-using-rpy2
        if ri.globalenv.find('dim')(r_obj) == ro.rinterface.NULL or np.array(value_rpy).ndim == 1:
            names = ri.globalenv.find('names')(r_obj)
            if names == ro.rinterface.NULL:
                value_py = pd.Series(
                    data=[v.item() for v in value_rpy],
                    index=range(len(value_rpy)),
                )
            else:
                value_py = pd.Series(
                    data=[v.item() for v in value_rpy],
                    index=list(names),
                )
        else:
            value_py = pd.DataFrame(
                data=value_rpy,
            )
            if ri.globalenv.find('rownames')(r_obj) != ro.rinterface.NULL:
                value_py.index = list(ro.r['rownames'](r_obj))
            if ri.globalenv.find('colnames')(r_obj) != ro.rinterface.NULL:
                value_py.columns = list(ri.globalenv.find('colnames')(r_obj))

        return value_py

    def __call__(self, r_script):
        try:
            return ro.r(r_script)
        except Exception as e:
            raise RuntimeError(f"R execution failed: {e}")

    def __repr__(self):
        var_infos = []
        for name in list(ri.globalenv):
            # value = ro.globalenv[name]
            value = ri.globalenv.find(name) # returns an rinterface-level object, no conversion yet
            
            # Try to determine shape/length, if possible
            try:
                if hasattr(value, "dim") and value.dim is not None:
                    shape = tuple(value.dim)
                elif hasattr(value, "ncol") and hasattr(value, "nrow"):
                    shape = (value.nrow, value.ncol)
                elif hasattr(value, "length"):
                    shape = (value.length, )
                else:
                    shape = (len(value), )
            except Exception:
                shape = None
            var_infos.append({
                'name': name,
                'type': type(value).__name__,
                'shape': shape,
            })
        var_infos = pd.DataFrame(var_infos)
        return str(var_infos)

    @classmethod
    def obj2cat(cls, dataframe: pd.DataFrame):
        converted = dataframe.copy()
        for col in converted.columns:
            if converted[col].dtype == object:
                converted[col] = converted[col].astype("category")
        return converted

    @classmethod
    def as_lines(cls, strings: list):
        return '\n'.join(map(str, strings))
    



