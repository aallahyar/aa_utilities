import pandas as pd


class RSpace():
    """A wrapper around `rpy2` package to facilitate import/export of variables between R and Python as well as running R commands.

    Returns:
        RSpace object: An instance of the RSpace wrapper

    Examples:

    """

    from rpy2 import robjects as ro
    from rpy2.robjects import pandas2ri

    def __init__(self, ipython=False):
        self.ro = RSpace.ro
        
        # loads IPython extension: https://rpy2.github.io/doc/latest/html/interactive.html#usage
        if ipython:
            %load_ext rpy2.ipython

    def __setitem__(self, name, value):
        with (self.ro.default_converter + RSpace.pandas2ri.converter).context():
            value_r = self.ro.conversion.get_conversion().py2rpy(value)
            self.ro.r.assign(name, value_r)

    def __getitem__(self, name):
        with (self.ro.default_converter + RSpace.pandas2ri.converter).context():
            value_r = self.ro.conversion.get_conversion().rpy2py(
                self.ro.globalenv[name])

        if isinstance(value_r, pd.DataFrame):
            return value_r
        value_df = pd.DataFrame(
            data=value_r,
        )

        # adding column names if present
        # source: https://stackoverflow.com/questions/12944250/handing-null-return-in-rpy2
        # source: https://stackoverflow.com/questions/73259425/how-to-load-a-rtypes-nilsxp-data-object-when-using-rpy2
        if self(f'names({name})') != self.ro.rinterface.NULL:
            value_df.index = list(self(f'names({name})'))
            value_df.columns = ['column']
        elif self(f'colnames({name})') != self.ro.rinterface.NULL:
            value_df.columns = self(f'colnames({name})')
        elif self(f'rownames({name})') != self.ro.rinterface.NULL:
            value_df.index = self(f'rownames({name})')

        return value_df

    def __call__(self, r_script):
        return self.ro.r(r_script)

    def __repr__(self):
        return f'''
        R space holds {len(self.ro.globalenv)} variables: {list(self.ro.globalenv)}
        '''

    @classmethod
    def obj2cat(cls, dataframe: pd.DataFrame):
        # print(dataframe.dtypes)
        converted = dataframe.copy()
        for col in converted.columns:
            if converted[col].dtype == object:
                converted[col] = converted[col].astype("category")
                # print(f'"{col}" dtype is now =>', data[col].dtype)
        # print(converted.dtypes)
        return converted

    @classmethod
    def as_lines(cls, strings: list):
        return '\n'.join(strings)


if __name__ == '__main__':
    R = RSpace()

    R("""
    data <- read.table(text="
    expression mouse treat1 treat2
    1 1.01 MOUSE1 NO NO
    2 1.04 MOUSE2 NO NO
    3 1.04 MOUSE3 NO NO
    4 1.99 MOUSE4 YES NO
    5 2.36 MOUSE5 YES NO
    6 2.00 MOUSE6 YES NO
    7 2.89 MOUSE7 NO YES
    8 3.12 MOUSE8 NO YES
    9 2.98 MOUSE9 NO YES
    10 5.00 MOUSE10 YES YES
    11 4.92 MOUSE11 YES YES
    12 4.78 MOUSE12 YES YES
    ", sep=" ", header=T)

    design <- model.matrix(~ treat1 + treat2, data=data)
    fit <- lm(formula='expression ~ treat1 + treat2', data=data)
    model_matrix <- model.matrix(fit)
    model_coef <- coef(fit)
    print(model_coef)
    """)
    print(R['model_matrix'])
    print(R['model_coef'])
