class Node:
    def __init__(self, id, left=None, right=None, merged=None, distance=0, clusters=None, parent=None):
        self.id = id
        self.left = left
        self.right = right
        if merged is None:
            self.merged = left.merged + right.merged
        else:
            self.merged = tuple(merged)
        self.distance = distance
        self.parent = parent
        self._clusters = clusters

    @property
    def clusters(self):
        return list(self._clusters.values())

    @property
    def is_leaf(self):
        return self.left is None

    @property
    def is_root(self):
        return self.parent is None

    @property
    def depth(self):
        d = 0
        node = self
        while node.parent is not None:
            d += 1
            node = node.parent
        return d

    @property
    def sibling(self):
        if self.parent is None:
            return None
        return self.parent.right if self.parent.left is self else self.parent.left

    def __repr__(self):
        clusters = self.clusters
        n_clusters = len(clusters)
        parent_id = f'<{self.parent.id}>' if self.parent else 'Empty'
        output = f'<Node {self.id}>: #clusters={n_clusters}, parent={parent_id}'
        for ci, cluster in enumerate(clusters[:5], 1):
            output += f'\n\tC{ci}: size={len(cluster)} | members={list(cluster[:5])}'
        return output


class LinkageTreeParser:
    """Parses the output of hierarchical clustering algorithms (scikit-learn or scipy) into a tree structure.
    Each node in the tree represents a cluster, with leaf nodes representing individual samples and internal
    nodes representing merged clusters. The parser provides easy access to cluster information,
    such as the members of each cluster, the distance at which clusters were merged, and the
    relationships between clusters (parent-child and sibling relationships).
    """

    def __init__(self, model):
        self.model = model
        if hasattr(model, 'children_'):  # scikit-learn
            merging_nodes = model.children_
            distances = model.distances_
            self.n_sample = len(model.labels_)
        else:  # scipy
            merging_nodes = model[:, :2].astype(int)
            distances = model[:, 2]
            self.n_sample = int(model[-1, 3])

        # initialize leaf nodes
        initial_clusters = {i: (i,) for i in range(self.n_sample)}
        self.tree = {}
        for node_id in range(self.n_sample):
            self.tree[node_id] = Node(
                id=node_id,
                merged=(node_id,),
                clusters=dict(initial_clusters),
            )

        # add internal nodes
        self._cut_map = {}  # n_clusters → node_id
        clusters = {i: (i,) for i in range(self.n_sample)}
        for (id_left, id_right), distance in zip(merging_nodes, distances):
            left = self.tree[id_left]
            right = self.tree[id_right]

            id_new = len(self.tree)
            clusters[id_new] = left.merged + right.merged
            del clusters[id_left], clusters[id_right]

            self.tree[id_new] = Node(
                id=id_new,
                left=left,
                right=right,
                distance=distance,
                clusters=dict(clusters),
            )
            self._cut_map[len(clusters)] = id_new

            left.parent = self.tree[id_new]
            right.parent = self.tree[id_new]

    @property
    def root(self):
        return self.tree[max(self.tree)]

    def cut(self, n_clusters):
        if not (1 <= n_clusters <= self.n_sample):
            raise ValueError(f'n_clusters must be between 1 and {self.n_sample}, got {n_clusters}')
        if n_clusters == self.n_sample:
            return {i: (i,) for i in range(self.n_sample)}
        return dict(self.tree[self._cut_map[n_clusters]]._clusters)

    def __repr__(self):
        output = f'Linkage tree: #nodes={len(self.tree)}'
        for ci, node in enumerate(reversed(list(self.tree.values())), 1):
            output += '\n' + repr(node)
            if ci >= 5:
                break
        return output

    def __getitem__(self, id):
        return self.tree[id]
