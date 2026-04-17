import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage

from aa_utilities.computation import LinkageTreeParser

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

# ── scikit-learn ──────────────────────────────────────────────────────────────
sk_model = AgglomerativeClustering(
    linkage='single',
    metric='euclidean',
    compute_distances=True,
).fit(X)

parsed = LinkageTreeParser(model=sk_model)

# tree overview: shows the top-5 nodes from root downward
print('=== Tree overview ===')
print(parsed)

# access any node directly by its id
print('\n=== Node access: parsed[2] ===')
print(parsed[2])

# root node
print('\n=== Root ===')
root = parsed.root
print(root)
print(f'  is_root  : {root.is_root}')
print(f'  is_leaf  : {root.is_leaf}')
print(f'  depth    : {root.depth}')
print(f'  distance : {root.distance}')
print(f'  sibling  : {root.sibling}')   # None — root has no sibling

# leaf node
print('\n=== Leaf node (id=0) ===')
leaf = parsed[0]
print(leaf)
print(f'  is_leaf  : {leaf.is_leaf}')
print(f'  is_root  : {leaf.is_root}')
print(f'  depth    : {leaf.depth}')
print(f'  sibling  : {leaf.sibling}')

# walk from a leaf up to the root via .parent
print('\n=== Path from leaf 0 to root ===')
node = parsed[0]
while node is not None:
    print(f'  Node {node.id}  distance={node.distance:.3f}  is_leaf={node.is_leaf}')
    node = node.parent

# cut the tree at a specific number of clusters
print('\n=== cut(n_clusters=2) ===')
clusters = parsed.cut(n_clusters=2)
for node_id, members in clusters.items():
    print(f'  cluster rooted at Node {node_id}: {members}')

print('\n=== cut(n_clusters=3) ===')
for node_id, members in parsed.cut(n_clusters=3).items():
    print(f'  cluster rooted at Node {node_id}: {members}')

# navigate a split: inspect both children of the root
print('\n=== Root split ===')
print(f'  left  child: Node {root.left.id}  members={root.left.merged}')
print(f'  right child: Node {root.right.id}  members={root.right.merged}')
print(f'  left.sibling is right: {root.left.sibling is root.right}')

# ── scipy ─────────────────────────────────────────────────────────────────────
print('\n=== scipy linkage matrix ===')
sp_model = linkage(
    X, 
    method='single', 
    metric='euclidean', 
    optimal_ordering=True, # only available in scipy, not available in scikit-learn
)
print(sp_model)

parsed_sp = LinkageTreeParser(model=sp_model)
print('\n=== scipy tree overview ===')
print(parsed_sp)

print('\n=== scipy cut(n_clusters=2) ===')
for node_id, members in parsed_sp.cut(n_clusters=2).items():
    print(f'  cluster rooted at Node {node_id}: {members}')

# plt.show(block=True)
# print()