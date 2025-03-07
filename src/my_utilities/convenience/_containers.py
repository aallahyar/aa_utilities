import numpy as np
import pandas as pd

class Container(pd.Series):
    """_summary_

    Args:
        pd (_type_): _description_
    
    Example:
        container = Container(
            data=dict(A=1000, B=10000), 
            series_kws=dict(name='NAMEEEE'), 
            repr_max_rows=14,
        )

        container['a'] = 2
        container.loc['b'] = 20.234

        container.c = 30 # NOTE: this is not going to be represented (but is available)! Implementation choise to protect typos
        print(container.c)

        container['d'] = pd.Series(dict(X=10, Y=1000))
        container['f'] = pd.DataFrame(dict(X=[10, 1000], Y=['asdf', 'asdaaaf']))
        container['Z'] = pd.DataFrame(np.random.rand(50, 50))

        container['e'] = 'MISTAKE1'
        container.e = 'CORRECT1'
        container['e1000'] = 'MISTAKE2'
        container['e1000'] = 'CORRECT2'
        container['e100'] = 'LAST ELEMENT'
        container['e100000'] = 'MISSING'

        print(container)

    """
    
    def __init__(self, data=None, repr_max_rows=20, repr_max_cols=80, series_kws=None, **kwargs):
        assert kwargs == {} # NOT IMPLEMENTED

        if series_kws is None:
            series_kws = {}
        series_kws = {'dtype': object} | series_kws # this avoids automatic conversion of all values if a when Series is <int> and gets converted to <float>

        super().__init__(
            data=data,
            **series_kws,
        )
        self.repr_max_cols = repr_max_cols
        self.repr_max_rows = repr_max_rows
        

    def __repr__(self):
        output = ''
        tab = ' ' * 4
        for row_num, (index, value) in enumerate(self.items(), start=1):
            match value:
                case pd.Series():
                    meta = f'<pd.Series> ({len(value)})'
                    preview = tab + f'\n{tab}'.join(line[:self.repr_max_cols] for line in str(value).split('\n')[:self.repr_max_rows])
                case pd.DataFrame():
                    meta = f'<pd.DataFrame> {value.shape}'
                    preview = tab + f'\n{tab}'.join(line[:self.repr_max_cols] for line in str(value).split('\n')[:self.repr_max_rows])
                case int(): #  | float()
                    meta = f'<int>'
                    preview = f'{tab}{str(value)[:self.repr_max_cols]}'
                case float():
                    meta = f'<float>'
                    preview = f'{tab}{str(value)[:self.repr_max_cols]}'
                case str():
                    meta = f'<str>'
                    preview = f'{tab}{str(value)[:self.repr_max_cols]}'
                case _:
                    meta = f'<{type(value)}>'
                    preview = tab + f'\n{tab}'.join(line[:self.repr_max_cols] for line in str(value).split('\n')[:self.repr_max_rows])
            output += (
                f'{index}: {meta}\n'
                f'{preview}\n'
            )
            if row_num >= self.repr_max_rows:
                output += '... MAXIMUM # ROWS REACHED ...\n'
                break
        
        # last row details
        if self.name is not None:
            output += f'Name: {self.name} | '
        output += f'#elements: {len(self):,d} | dtype: {self.dtype}\n'

        return output
        

if __name__ == '__main__':
    container = Container(
        data=dict(A=1000, B=10000), 
        series_kws=dict(name='NAMEEEE'), 
        repr_max_rows=14,
    )

    container['a'] = 2
    container.loc['b'] = 20.234

    container.c = 30 # NOTE: this is not going to be represented (but is available)! Implementation choise to protect typos
    print(container.c)

    container['d'] = pd.Series(dict(X=10, Y=1000))
    container['f'] = pd.DataFrame(dict(X=[10, 1000], Y=['asdf', 'asdaaaf']))
    container['Z'] = pd.DataFrame(np.random.rand(50, 50))

    container['e'] = 'MISTAKE1'
    container.e = 'CORRECT1'
    container['e1000'] = 'MISTAKE2'
    container['e1000'] = 'CORRECT2'
    container['e100'] = 'LAST ELEMENT'
    container['e100000'] = 'MISSING'

    print(container)