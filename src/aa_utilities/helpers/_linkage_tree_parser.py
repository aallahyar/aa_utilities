
class Node:
    def __init__(self, id, left=None, right=None, merged=None, distance=0, clusters=None, parent=None):
        self.id = id
        self.left = left
        self.right = right
        self.merged = merged
        if merged is None:
            self.merged = left.merged + right.merged
        self.distance = distance
        self.parent = parent
        self._clusters = clusters
    
    @property
    def clusters(self):
        return list(self._clusters.values())
    
    def __repr__(self):
        n_clusters = len(self.clusters)
        parent_id = f'<{self.parent.id}>' if self.parent else 'Empty'
        output = f'<Node {self.id}>: #clusters={n_clusters}, parent={parent_id}'
        for ci in range(min(n_clusters, 5)):
            cluster = self.clusters[ci]
            output += f'\n\tC{ci+1}: size={len(cluster)} | members={cluster[:5]}'
        return output


class LinkageTreeParser():
    """
    Example:
    """

    def __init__(self, model):
        self.model = model
        if hasattr(model, 'children_'): # its scikit-learn
            merging_nodes = model.children_
            distances = model.distances_
            self.n_sample = len(model.labels_)
        else: # its scipy
            merging_nodes = model[:, :2].astype(int)
            distances = model[:, 2]
            self.n_sample = int(model[-1, 3])

        # initialize tree with leaf nodes
        clusters = {i: [i] for i in range(self.n_sample)}
        self.tree = {}
        for id_leaf in range(self.n_sample):
            self.tree[id_leaf] = Node(
                id=id_leaf,
                merged=[id_leaf], 
                clusters=clusters.copy(),
            )

        # add non-leaf nodes
        for (id_left, id_right), distance in zip(merging_nodes, distances):
            left = self.tree[id_left]
            right = self.tree[id_right]
            
            # add new node
            id_new = len(self.tree)
            self.tree[id_new] = Node(
                id=id_new,
                left=left,
                right=right,
                distance=distance,
            )

            # update clusters
            clusters[id_new] = self.tree[id_new].merged
            del clusters[id_left], clusters[id_right]
            self.tree[id_new]._clusters = clusters.copy()

            # update parents
            left.parent = self.tree[id_new]
            right.parent = self.tree[id_new]
    
    def __repr__(self):
        output = f'Linkage tree: #nodes={len(self.tree)}'
        for ci, child in enumerate(reversed(self.tree.values()), 1):
            output += '\n' + repr(child)
            if ci >= 5:
                break
        return output
    
    def __getitem__(self, id):
        return self.tree[id]


if __name__ == '__main__':
    import pandas as pd
    from sklearn.cluster import AgglomerativeClustering

    import aa_utilities

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
    
    parsed = aa_utilities.convenience.LinkageTreeParser(model=model)
    print(parsed)
    print(parsed[2])

    # scipy example
    from scipy.cluster.hierarchy import linkage
    model = linkage(X, optimal_ordering=True, method='single', metric='euclidean')
    print(model)
    parsed = aa_utilities.convenience.LinkageTreeParser(model=model)
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

