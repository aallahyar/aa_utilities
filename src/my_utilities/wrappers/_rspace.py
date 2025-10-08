import numpy as np
import pandas as pd
from rpy2 import robjects as ro
from rpy2.robjects import pandas2ri
import rpy2.rlike.container as rlc

class RSpace():
    """A wrapper around `rpy2` package to facilitate import/export of variables between R and Python as well as running R commands.

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
            value = rlc.TaggedList(value.values(), tags=value.keys())

        with (ro.default_converter + pandas2ri.converter).context():
            value_r = ro.conversion.get_conversion().py2rpy(value)
            ro.r.assign(name, value_r)

    def __getitem__(self, name):
        with (ro.default_converter + pandas2ri.converter).context():
            value_rpy = ro.conversion.get_conversion().rpy2py(ro.globalenv[name])
        
        # performing type conversions
        r_obj = ro.globalenv[name]
        
        # check if the variable is scalar: https://stackoverflow.com/questions/38088392/how-do-you-check-for-a-scalar-in-r
        if ro.r['is.atomic'](r_obj)[0] and ro.r['length'](r_obj)[0] == 1:
            if isinstance(value_rpy, (list, tuple, np.ndarray, )) and len(value_rpy) == 1:
                # assert len(value_rpy) == 1, f'Unexpected length for a scalar variable: {value_rpy}'
                return value_rpy[0]
            else:
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
        if ro.r['dim'](r_obj) == ro.rinterface.NULL:
            names = ro.r['names'](r_obj)
            if names == ro.rinterface.NULL:
                value_py = pd.Series(
                    data=list(value_rpy),
                    index=range(len(value_rpy)),
                )
            else:
                value_py = pd.Series(
                    data=value_rpy,
                    index=list(names),
                )
        else:
            value_py = pd.DataFrame(
                data=value_rpy,
            )
            if ro.r['rownames'](r_obj) != ro.rinterface.NULL:
                value_py.index = list(ro.r['rownames'](r_obj))
            if ro.r['colnames'](r_obj) != ro.rinterface.NULL:
                value_py.columns = list(ro.r['colnames'](r_obj))

        return value_py

    def __call__(self, r_script):
        return ro.r(r_script)

    def __repr__(self):
        var_infos = []
        for name in list(ro.globalenv):
            value = ro.globalenv[name]
            
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


