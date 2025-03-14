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

        # comparing containers
        c1 = container.copy()
        print(container == c1)
        c1['g9'] = 'teasdf'
        print(container == c1)
        print(container == {})

        # 30
        # Name: NAMEEEE | #elements: 12 | dtype: object
        # A1: <int>
        #     1000
        # B2: <int>
        #     10000
        # a3: <int>
        #     2
        # b4: <float>
        #     20.234
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
        #     0   0.997509  0.912642  0.772244  0.952102  0.887265  0.163814  0.394557  ...  0
        #     1   0.353464  0.606520  0.867093  0.041485  0.955350  0.644399  0.467551  ...  0
        #     2   0.643037  0.225938  0.962445  0.897610  0.726765  0.330237  0.819648  ...  0
        #     3   0.247031  0.275309  0.146851  0.243915  0.528953  0.499110  0.663551  ...  0
        #     4   0.759634  0.849217  0.633940  0.328621  0.466124  0.803929  0.011944  ...  0
        #     5   0.097385  0.904525  0.532073  0.157808  0.166286  0.956699  0.607373  ...  0
        #     6   0.070963  0.400034  0.529811  0.212139  0.260430  0.196273  0.440619  ...  0
        #     7   0.805794  0.748254  0.489292  0.853960  0.550034  0.685125  0.997672  ...  0
        #     8   0.209074  0.932858  0.396076  0.033766  0.522982  0.173776  0.553461  ...  0
        #     9   0.756171  0.000294  0.070033  0.353588  0.052346  0.190776  0.719303  ...  0
        #     10  0.705756  0.869959  0.080893  0.615639  0.162093  0.644244  0.916912  ...  0
        #     11  0.010183  0.624911  0.743288  0.351793  0.184802  0.480907  0.305374  ...  0
        #     12  0.189890  0.470661  0.686508  0.993452  0.043443  0.105803  0.858113  ...  0
        #     13  0.830304  0.445530  0.189642  0.589793  0.828023  0.517355  0.455617  ...  0
        #     14  0.331153  0.122731  0.719543  0.845514  0.503989  0.042748  0.771945  ...  0
        #     15  0.523536  0.433391  0.288636  0.638947  0.641904  0.379454  0.090701  ...  0
        #     16  0.308179  0.746568  0.304162  0.555527  0.993880  0.659259  0.340334  ...  0
        #     17  0.378564  0.635957  0.323096  0.332257  0.109309  0.567347  0.806181  ...  0
        #     18  0.488180  0.120617  0.439384  0.604643  0.765385  0.847203  0.005361  ...  0
        # g9: <str>
        #     test1
        # i10: <str>
        #     Corrected2
        # k11: <str>
        #     LAST ELEMENT
        # ... ...

        

    """
    
    def __init__(self, data=None, series_kws=None, repr_max_n_elements=200, repr_max_cols=80, prev_max_rows=20, **kwargs):
        assert kwargs == {} # NOT IMPLEMENTED

        if series_kws is None:
            series_kws = {}
        series_kws = {'dtype': object} | series_kws # this avoids automatic conversion of all values if a when Series is <int> and gets converted to <float>

        super().__init__(
            data=data,
            **series_kws,
        )
        self.repr_max_cols = repr_max_cols
        self.repr_max_n_elements = repr_max_n_elements
        self.prev_max_rows = prev_max_rows
        

    def __repr__(self):
        tab = ' ' * 4
        output = ''

        # add details in the first row 
        if self.name is not None:
            output += f'Name: {self.name} | '
        output += f'#elements: {len(self):,d} | dtype: {self.dtype}\n'
        
        # add element details
        for row_num, (index, value) in enumerate(self.items(), start=1):
            match value:
                case pd.Series():
                    meta = f'<pd.Series> ({len(value)})'
                    preview = tab + f'\n{tab}'.join(line[:self.repr_max_cols] for line in str(value).split('\n')[:self.prev_max_rows])
                case pd.DataFrame():
                    meta = f'<pd.DataFrame> {value.shape}'
                    preview = tab + f'\n{tab}'.join(line[:self.repr_max_cols] for line in str(value).split('\n')[:self.prev_max_rows])
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
                    preview = tab + f'\n{tab}'.join(line[:self.repr_max_cols] for line in str(value).split('\n')[:self.prev_max_rows])
            output += (
                f'{index}: {meta}\n'
                f'{preview}\n'
            )
            if row_num >= self.repr_max_n_elements:
                output += '... ...\n'
                break

        return output
    
    def __eq__(self, other):
        if not isinstance(other, pd.Series):
            raise ValueError('Can only compare `Container` or `pd.Series()` instances.')
        return self.equals(other)
        

if __name__ == '__main__':
    container = Container(
        data=dict(A1=1000, B2=10000), 
        series_kws=dict(name='NAMEEEE'), 
        prev_max_rows=20,
        repr_max_n_elements=11,
    )

    container['a3'] = 2
    container.loc['b4'] = 20.234

    container.c5miss = 30 # NOTE: this is not going to be represented (but is available)! Implementation choise to protect typos
    print(container.c5miss)

    container['c5show'] = 300

    container['d6'] = pd.Series(dict(X=10, Y=1000))
    container['e7'] = pd.DataFrame(dict(X=[10, 1000], Y=['asdf', 'asdaaaf']))
    container['f8'] = pd.DataFrame(np.random.rand(50, 50))

    container['g9'] = 'test1'
    container['i10'] = 'MISTAKE2'
    container.i10 = 'Corrected2'
    container['k11'] = 'LAST ELEMENT'
    container['l12'] = 'Not shown'

    print(container)
    c1 = container.copy()
    print(container == c1)
    c1['g9'] = 'teasdf'
    print(container == c1)
    print(container == {})

