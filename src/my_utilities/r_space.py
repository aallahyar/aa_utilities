import pandas as pd

class RSpace():
    from rpy2 import robjects as ro
    from rpy2.robjects import pandas2ri

    __version__ = '0.0.1'
    
    def __init__(self):
        self.ro = RSpace.ro
    
    def __setitem__(self, name, value):
        with (self.ro.default_converter + RSpace.pandas2ri.converter).context():
            value_r = self.ro.conversion.get_conversion().py2rpy(value)
            self.ro.r.assign(name, value_r)
    
    def __getitem__(self, name):
        with (self.ro.default_converter + RSpace.pandas2ri.converter).context():
            valur_r = self.ro.conversion.get_conversion().rpy2py(self.ro.globalenv[name])
        return valur_r
    
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