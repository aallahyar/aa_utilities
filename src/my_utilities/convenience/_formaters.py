import sys
import pprint

import numpy as np
import pandas as pd


class PrettyPrinter():
    def __init__(
            self, 
            indent=' ' * 4, 
            display_width=120, 
            max_n_rows=100,
            max_n_elements=20,
        ):
        self.indent = indent
        self.pp = pprint.PrettyPrinter()
        self.display_width = display_width
        self.max_n_rows = max_n_rows
        self.max_n_elements = max_n_elements
    
    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            assert hasattr(self, key), 'Unknown PrettyPrinter parameter'
            setattr(self, key, value)

    def clip(self, representation, indent=None):
        outputs = []
        lines = representation.split('\n')
        for line in lines[:self.max_n_rows]:
            output = line[:self.display_width]
            if len(line) > self.display_width:
                output += ' ...'
            outputs.append(output)
        if len(lines) > self.max_n_rows:
                outputs.append('...')
        if indent:
            return f'\n{indent}'.join(outputs)
        else:
            return f'\n{self.indent}'.join(outputs)
    
    def pformat(self, obj):
        meta = f'<{obj.__class__.__name__}>'
        if isinstance(obj, (dict, list, tuple, set, pd.Series)):
            meta += f' ({len(obj)})'
        if isinstance(obj, (pd.DataFrame, )):
            meta += f' {obj.shape}'
        
        match obj:
            case int() | float() | str():
                preview = f'{obj!r}'
            case dict() | list() | tuple() | set() if type(obj) in [dict, list, tuple, set]:

                # define boundaries
                if isinstance(obj, (dict, set)):
                    bnds = '{}'
                if isinstance(obj, (list, )):
                    bnds = '[]'
                if isinstance(obj, (tuple, )):
                    bnds = '()'

                outputs = [f'{bnds[0]}']
                idx_ndigit = np.log10(len(obj)).astype(int) + 1
                for idx, item in enumerate(obj):
                    if isinstance(obj, (dict, )):
                        key_repr = self.clip(self.pformat(item))
                        value_repr = self.clip(self.pformat(obj[item]))
                        outputs.append(f'{key_repr}: {value_repr},')
                    else:
                        value_repr = self.clip(self.pformat(item))
                        if isinstance(obj, (list, )):
                            outputs.append(f"{f'[{idx}]':>{idx_ndigit + 2}s} {value_repr},")
                        else:
                            outputs.append(f'{value_repr},')
                    if idx + 1 == self.max_n_elements:
                        outputs.append('...')
                        break
                # outputs.append(f'{bnds[1]}')
                preview = f'{meta} ' + f'\n{self.indent}'.join(outputs) + f'\n{bnds[1]}'
            # case Container():
            #     preview = self.clip(f'{meta}\n{obj}')
            case pd.Series():
                preview = self.clip(f'{meta}\n{obj}')
            case pd.DataFrame():
                preview = self.clip(f'{meta}\n{obj}')
            case _:
                preview = f'{obj!r}'

        return preview

if __name__ == '__main__':
    if sys.path[0] != './':
        sys.path.insert(0, './')

    pp = PrettyPrinter()

    obj = {f'key{v}': v for v in range(10)}
    obj[1] = {'a': 1, 2: 'b'}
    obj['a3'] = {0: 1, 'str': 'string', 'list': ['a', 'b', 1, 2], 'dict': {'a': 1, 2: 'b'}, 'tuple': (
        'a', 'b', 1, 2), 'function': lambda x: x, 'unicode': u'\xa7', ("tuple", "key"): "valid-"*20}
    # obj['g9'] = Container(
    #     g9a=[12, 100],
    #     g9b=list('abcdefghijklmopqrstuvwxyz'),
    #     g9c=12,
    # )
    obj['d6'] = pd.Series(dict(X=10, Y=1000))
    obj['e7'] = pd.DataFrame(dict(X=[10, 1000], Y=['asdf', 'asdaaaf']))
    obj['f8'] = pd.DataFrame(np.random.rand(50, 50))
    print(pp.pformat(obj))

