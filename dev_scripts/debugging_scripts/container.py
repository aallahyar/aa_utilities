import numpy as np
import pandas as pd

from aa_utilities.storage import Container

s = pd.Series(dict(X=10, Y=1000))

container = Container(
    A1=1000, 
    get=[10000, 'dict.get() is no longer accessible'],
    # values=['test', 'test1'], # with this, conversion with pd.Series() fails
)
print(container)
# container.set_params(
#     # prev_max_rows=30,
#     repr_max_n_elements=11,
# )
print(container.get)
print(container.values)

container['a3'] = {0: 1, 'str': 'string', 'list': ['a', 'b', 1, 2], 'dict': {'a': 1, 2: 'b'}, 'tuple': (
    'a', 'b', 1, 2), 'function': lambda x: x, 'unicode': u'\xa7', ("tuple", "key"): "valid-"*20}
container.get[0] = 20.234
container.update(A1=21, b4=21.234)
container.c5 = [300, 200.3, 'dot notation works!']
print(container)
print(container.keys())

container['d6'] = pd.Series(dict(X=10, Y=1000))
container['e7'] = pd.DataFrame(dict(X=[10, 1000], Y=['asdf', 'asdaaaf']))
container['f8'] = pd.DataFrame(np.random.rand(50, 50))

container['g9'] = Container(
    g9a=[12, 100],
    g9b=list('abcdefghijklmopqrstuvwxyz'),
    g9c={f'key{k}': f'value={k}' for k in 'abcdefghijklmopqrstuvwxyz'},
    g9d=set('abcdefghijklmopqrstuvwxyz'),
)
container.g9._pp.set_params(max_n_elements=11)

container['i10'] = 'MISTAKE'
container.i10 = 'Corrected'

container._hidden = 'this is available, but stays hidden'
print(container._hidden)

container.k11 = 'LAST ELEMENT'
container['l12'] = 'Not shown'

print(container)

c1 = pd.Series(container).copy()
print(container == c1)
c1.g9 = 'not test1 anymore'
print(c1['g9'])
print(container == c1)
print(container == c1.to_dict())
print(container == {})

(
    pd.Series(container)
    .filter(items=['g9'])
    .pipe(Container.from_series)
    # .copy()
)

(
    container
    .drop(index=['A1', 'get', 'a3'])
)