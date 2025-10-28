
import numpy as np
import pandas as pd


def code_natsorted(srs):
    from natsort import index_natsorted

    if not isinstance(srs, (pd.core.series.Series, )):
        srs = pd.Series(srs)

    natsort_idx = np.argsort(index_natsorted(srs))
    return srs.groupby(by=srs).transform(lambda grp: natsort_idx[grp.index].min())

if __name__ == '__main__':
    # import numpy as np
    import pandas as pd

    df = pd.DataFrame({
        'col1': ['A', 'B', 'A', 'A', 'A'],
        'col2': ['W1', 'W3', 'W17', 'W11', 'W2'],
    })
    print(df)
    #   col1 col2
    # 0    A   W1
    # 1    B   W3
    # 2    A  W17
    # 3    A  W11
    # 4    A   W2

    print(df.sort_values(by=['col1', 'col2']))
    #   col1 col2
    # 0    A   W1
    # 2    A  W11
    # 3    A  W17
    # 4    A   W2
    # 1    B   W3

    print(df.sort_values(by=['col1', 'col2'], key=code_natsorted))
    #   col1 col2
    # 0    A   W1
    # 4    A   W2
    # 2    A  W11
    # 3    A  W17
    # 1    B   W3