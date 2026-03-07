
from numba import jit as _jit
import numpy as _np


# source: https://stackoverflow.com/questions/52371329
@_jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def mean(mat):
    n = len(mat)
    b = _np.empty(n)
    for i in range(n):
        b[i] = mat[i].mean()
    return b

@_jit(nopython=True)
def std(mat):
    n = len(mat)
    b = _np.empty(n)
    for i in range(n):
        b[i] = mat[i].std()
    return b

@_jit(nopython=True)
def rank(mat):
    i, j = _np.meshgrid(*map(_np.arange, mat.shape), indexing='ij')
    s = mat.argsort(1)
    out = _np.empty_like(s)
    out[i, s] = j
    return out

@_jit(nopython=True)
def _pearson_numba(A, B, low_memory=False):
    """ Pearson's correlation """
    
    n, k = A.shape
    m, k = B.shape

    mu_a = mean(A)
    mu_b = mean(B)
    sig_a = std(A)
    sig_b = std(B)

    if low_memory:
        out = _np.empty((n, m))
        for i in range(n):
            for j in range(m):
                out[i, j] = (A[i] - mu_a[i]) @ (B[j] - mu_b[j]) / (k * sig_a[i] * sig_b[j])
        return out
    else:    
        return (A - mu_a[:, None]) @ (B - mu_b[:, None]).T / (k * sig_a * sig_b[:, None])
    

# source: https://stackoverflow.com/questions/71844846
def _pearson_numpy(A, B):
    am = A - _np.mean(A, axis=1, keepdims=True)
    bm = B - _np.mean(B, axis=1, keepdims=True)
    return am @ bm.T /  (
        _np.sqrt(_np.sum(am**2, axis=1, keepdims=True)).T * 
        _np.sqrt(_np.sum(bm**2, axis=1, keepdims=True))
    )

def spearman(A, B=None, low_memory=False):
    if B is None:
        return _pearson_numba(rank(A), rank(A), low_memory=low_memory)
    else:
        return _pearson_numba(rank(A), rank(B), low_memory=low_memory)
        

if __name__ == '__main__':
    from scipy import stats
    import pandas as pd
    import timeit

    _np.random.seed([3, 1415])
    A = _np.random.randn(200, 1000)
    B = _np.random.randn(200, 1000)

    funcs = {
        'numba_lm0': lambda: _pearson_numba(A, A, low_memory=False),
        'numba_lm1': lambda: _pearson_numba(A, A, low_memory=True),
        'my_numpy':  lambda: _pearson_numpy(A, A),
        'np_coef':   lambda: _np.corrcoef(A),
        'pd_coef':   lambda: pd.DataFrame(A.T).corr(method='pearson'),
        'scipy_coef':   lambda: stats.pearsonr(A.T).statistic,  # only supports 1D arrays
    }

    for i, (name, func) in enumerate(funcs.items()):
        if i == 0:
            res = func()
        else:
            assert _np.allclose(res, func())
        elps_time = timeit.timeit(func, number=100)
        print(f'{name:10}: {elps_time:0.5f}')

# output:
# numba_lm0 : 3.33725
# numba_lm1 : 3.28394
# my_numpy  : 2.61974
# np_coef   : 0.31122
# pd_coef   : 6.64400