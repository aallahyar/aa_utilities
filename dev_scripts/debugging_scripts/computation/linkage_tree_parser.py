import pandas as pd
from sklearn.cluster import AgglomerativeClustering

import aa_utilities.computation

X = pd.DataFrame(
    data=[
        [1, 1],
        [1, 4],
        [1, 2],
        [3, 1],
        [4, 1],
    ],
    columns=['x1', 'x2'],
    index=list('abcde'),
)
print(X)

# scikit-learn example
model = AgglomerativeClustering(
    linkage='single', 
    metric='euclidean',
    compute_distances=True,
).fit(X)
print(model)

parsed = aa_utilities.computation.LinkageTreeParser(model=model)
print(parsed)
print(parsed[2])
print('root:', parsed.root)
print('root.is_root:', parsed.root.is_root)
print('root.is_leaf:', parsed.root.is_leaf)
print('leaf 0 depth:', parsed[0].depth)
print('leaf 0 sibling:', parsed[0].sibling)
print('cut n=2:', parsed.cut(n_clusters=2))

# scipy example
from scipy.cluster.hierarchy import linkage
model = linkage(X, optimal_ordering=True, method='single', metric='euclidean')
print(model)
parsed = aa_utilities.computation.LinkageTreeParser(model=model)
print(parsed)

# from matplotlib import pyplot as plt
# import numpy as np
# import seaborn as sns
# cls_map = sns.clustermap(
#     data=X,
#     row_linkage=model,
#     cmap="vlag",
#     # colors_ratio = 0.02,
#     center=0,
#     # vmin=-3, vmax=3,
#     # cbar_pos=(1.0, 0.7, 0.02, 0.2),
#     figsize=(4, 7),
# )

# dgram = cls_map.dendrogram_row.dendrogram
# I = np.array(dgram['icoord'])
# D = np.array(dgram['dcoord'])

# plt.show(block=True)
# print()