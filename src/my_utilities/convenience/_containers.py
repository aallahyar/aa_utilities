import sys
from copy import deepcopy

import numpy as np
import pandas as pd

from ..convenience import PrettyPrinter
from ._loggers import setup_logger
from .._configurations import configs


class Container(dict):
    """_summary_

    Args:
        pd (_type_): _description_
    
    Example:
        container = Container(
            A1=1000, 
            B2=10000,
        )
        container._pp.set_params(
            display_width=120,
            prev_max_rows=20,
            repr_max_n_elements=11,
        )

        container['a3'] = 20
        container.loc['b4'] = 20.234
        container.include(a3=21, b4=21.234)

        container.c5miss = 30 # NOTE: this is not going to be represented (but is available)! Implementation choise to protect typos
        print(container.c5miss)

        container['c5show'] = [300, 200.3]

        container.drop(regex=r'[4-9]$')

        container.filter(regex=r'[1-3]$')

        # 30
        # Name: NAMEEEE | #elements: 12 | dtype: object
        # A1: <int>
        #     1000
        # B2: <int>
        #     10000
        # a3: <int>
        #     21
        # b4: <float>
        #     21.234
        # c5show: <int>
        #     300
        # d6: <pd.Series> (2)
        #     X      10
        #     Y    1000
        #     dtype: int64
        # e7: <pd.DataFrame> (2, 2)
        #         X        Y
        #     0    10     asdf
        #     1  1000  asdaaaf
        # f8: <pd.DataFrame> (50, 50)
        #             0         1         2         3         4         5         6   ...   
        #     0   0.002107  0.338896  0.690741  0.812407  0.963908  0.973091  0.471101  ...  0
        #     1   0.770973  0.562469  0.126399  0.598199  0.434090  0.809549  0.218046  ...  0
        #     2   0.878720  0.822819  0.687376  0.363813  0.437782  0.578828  0.903076  ...  0
        #     3   0.627770  0.751263  0.136631  0.160570  0.763360  0.316500  0.683757  ...  0
        #     4   0.006530  0.251790  0.511888  0.739777  0.290485  0.790197  0.425332  ...  0
        #     5   0.370521  0.655337  0.990668  0.816418  0.446793  0.346782  0.950491  ...  0
        #     6   0.730079  0.058572  0.793605  0.102819  0.539840  0.589176  0.828790  ...  0
        #     7   0.023172  0.823954  0.094029  0.113598  0.766048  0.989696  0.249909  ...  0
        #     8   0.446134  0.654500  0.636310  0.014290  0.357877  0.331157  0.750483  ...  0
        #     9   0.188808  0.316420  0.009136  0.431169  0.893057  0.930005  0.677143  ...  0
        #     10  0.832135  0.357086  0.803372  0.502375  0.101918  0.157935  0.001590  ...  0
        #     11  0.488486  0.300547  0.073376  0.052409  0.262669  0.369836  0.215307  ...  0
        #     12  0.092946  0.165360  0.534842  0.009376  0.629057  0.717681  0.861316  ...  0
        #     13  0.800748  0.781123  0.397608  0.698807  0.616372  0.016152  0.476463  ...  0
        #     14  0.925485  0.824469  0.552669  0.393818  0.692628  0.903807  0.564311  ...  0
        #     15  0.454299  0.507266  0.958788  0.754197  0.929267  0.561252  0.988922  ...  0
        #     16  0.212193  0.319385  0.698605  0.045540  0.798270  0.931568  0.515912  ...  0
        #     17  0.471345  0.783311  0.961724  0.220723  0.323423  0.255226  0.046600  ...  0
        #     18  0.826631  0.409814  0.503957  0.001145  0.922952  0.231114  0.054092  ...  0
        # g9: <str>
        #     test1
        # i10: <str>
        #     Corrected2
        # k11: <str>
        #     LAST ELEMENT
        # ... ...

        # True
        # A1                                                     1000
        # B2                                                    10000
        # a3                                                       21
        # b4                                                   21.234
        # c5show                                                  300

        # not test1 anymore
        # False
        # Traceback (most recent call last):
        # ....
        # ValueError: Can only compare `Container` or `pd.Series()` instances.

        

    """
    
    def __init__(self, **kwargs):
        # initialization: Set up base infrastructure (private attributes, not user data)
        # Set _RESERVED_TERMS and logger before calling super().__init__ to avoid recursion in __setattr__
        # specifically, it directly calls the parent class’s version of __setattr__.
        # and skips any custom logic implemented in Container.__setattr__.
        # The attribute is stored in the instance’s __dict__ as usual.
        ## Get all public attributes/methods (exclude dunder and private by convention)
        super().__setattr__('_pp', PrettyPrinter())
        super().__setattr__('_params', PrettyPrinter())
        super().__setattr__('_logger', setup_logger(name=__name__, level=configs.log.level))
        super().__setattr__('_RESERVED_TERMS', {
            key for key in dir(type(self))
            if not key.startswith('__') and not key.startswith('_')
        })

        # configuration: handle user data logic in `_configure`
        self._configure(**kwargs)

    def _configure(self, **kwargs):
        """Separation of instantiation from configuration; add user data here."""
        for key, value in kwargs.items():
            self.__setattr__(key, value)
    
    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            assert key in self._params, 'Unknown parameter'
            self._params[key] = value
    
    def copy(self, deep=True):
        if deep:
            # Do not use pd.Series.copy(): 
            # When copying an object containing Python objects, a deep copy will copy the data, 
            # but will not do so recursively. Updating a nested data object will be reflected in the deep copy.
            # https://pandas.pydata.org/docs/reference/api/pandas.Series.copy.html
            return deepcopy(self)
        else:
            return Container(**self)
    
    @classmethod
    def from_series(cls, series):
        # or equally: Container(**series)
        # or equally: Container(**{'A': [1,2,3]})
        return cls(**series)
    
    def to_series(self):
        return pd.Series(self)

    def to_dict(self):
        return dict(self)

    def drop(self, regex=None, **kwargs):
        srs = self.to_series()
        if regex is None:
            result = srs.drop(**kwargs)
        else:
            labels = srs.filter(regex=regex).keys()
            result = srs.drop(labels=labels, **kwargs)
        return self.from_series(result)

    def filter(self, **kwargs):
        return self.from_series(self.to_series().filter(**kwargs))
    
    def __getattribute__(self, name):
        if name in self:
            return self[name]
        return super().__getattribute__(name)
    
    def __setattr__(self, name, value):
        if name in self._RESERVED_TERMS:
            self._logger.warning(
                f'Setting "{name}" will shadow the existing attribute (or method) with the same name'
                f'that exists in this <{type(self).__name__}>.'
            )
        if name.startswith('_'):
            super().__setattr__(name, value) # would not appear in .items(); TODO: not very useful it seems. Need more time to figure out if this is useful
        else:
            self[name] = value

    def __repr__(self):        
        
        # add details for each element
        output = f'<Container> ({len(self):d})'
        for row_num, (index, value) in enumerate(self.items(), start=1):
            
            # define connector
            if row_num == len(self) or row_num >= self._pp.max_n_elements:
                connector = '└─■'
                indent = ' ' + ' ' * 3
            else:
                connector = '├─■'
                indent = '│' + ' ' * 3

            # prepare preview
            preview = self._pp.clip(self._pp.pformat(value), indent=indent)
            output += f'\n{connector} {index}: {preview}'

            if row_num >= self._pp.max_n_elements:
                output += '\n... ...'
                break

        return output
    
    def __eq__(self, other):
        if isinstance(other, (pd.Series, dict, Container)):
            return dict(self) == dict(other)
        else:
            return False
            # raise NotImplementedError(
            #     'Can only compare `Container`, `pd.Series()` instance.'
            #     f' Got {type(other)} instead.'
            #     )





    
