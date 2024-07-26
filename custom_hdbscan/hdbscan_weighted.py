import numpy as np
from collections import deque
from custom_hdbscan.unionfind import UnionFind, TreeUnionFind

class WeightedHDBSCAN:
    def __init__(self, min_cluster_weight=5):
        self.min_cluster_weight = min_cluster_weight

    def _bfs_from_hierarchy(self, hierarchy, root):
        num_points = (hierarchy.shape[0] + 1) // 2 
        to_process = [root]
        bfs = []
        while to_process:
            bfs.extend(to_process)
            to_process = hierarchy[to_process,:2].flatten()
            to_process = to_process[to_process != np.array(None)].astype(int).tolist()
        return bfs

    def _bfs_from_cluster_tree(self, tree, root):
        to_process = [root]
        bfs = []
        while to_process:
            bfs.extend(to_process)
            to_process = tree['child'][np.isin(tree['parent'], to_process)].tolist()
        return bfs

    def generate_hierarchy(self, mst, node_weights, node_radii):
        sorted_mst = mst[np.argsort(mst.T[2])]  # Sort by weight (distance)

        U = UnionFind(node_weights)
        hierarchy = [(None, None, radius, weight) for radius, weight in zip(node_radii, node_weights)]
        for i, j, d in sorted_mst:
            i, j = int(i), int(j)
            root_i, root_j = U.find(i), U.find(j)
            root_union = U.union(root_i, root_j)
            hierarchy.append((root_i, root_j, d, U.weight[root_union]))

        return np.array(hierarchy)

    def condense_tree(self, hierarchy):
        num_nodes = hierarchy.shape[0]
        root = num_nodes - 1
        num_points = root // 2 + 1
        next_label = num_points + 1

        relabel = {root: num_points}
        visited = np.zeros(num_nodes, dtype=bool)
        result_list = []

        for node in self._bfs_from_hierarchy(hierarchy, root):
            if visited[node]:
                continue

            left, right, dist, size = hierarchy[node]
            lambda_value = 1 / dist if dist > 0 else np.inf

            if node < num_points:
                assert size >= self.min_cluster_weight  # TODO: Remove this later, sanity check for now
                result_list.append((relabel[node], node, lambda_value, size))
                continue

            left_weight = hierarchy[left][3]
            right_weight = hierarchy[right][3]

            if left_weight >= self.min_cluster_weight and right_weight >= self.min_cluster_weight:
                relabel[left] = next_label
                next_label += 1
                result_list.append((relabel[node], relabel[left], lambda_value, left_weight))

                relabel[right] = next_label
                next_label += 1
                result_list.append((relabel[node], relabel[right], lambda_value, right_weight))

            elif left_weight < self.min_cluster_weight and right_weight < self.min_cluster_weight:
                for subnode in self._bfs_from_hierarchy(hierarchy, left):
                    if subnode < num_points:
                        result_list.append((relabel[node], subnode, lambda_value, hierarchy[subnode][3]))
                    visited[subnode] = True

                for subnode in self._bfs_from_hierarchy(hierarchy, right):
                    if subnode < num_points:
                        result_list.append((relabel[node], subnode, lambda_value, hierarchy[subnode][3]))
                    visited[subnode] = True
                    
            else:   # Only one of the children is below the threshold
                dropped, retained = (left,right) if left_weight < self.min_cluster_weight else (right,left)
                relabel[retained] = relabel[node]
                for subnode in self._bfs_from_hierarchy(hierarchy, dropped):
                    if subnode < num_points:
                        result_list.append((relabel[node], subnode, lambda_value, hierarchy[subnode][3]))
                    visited[subnode] = True

        return np.array(result_list, dtype=[('parent', int), 
                                            ('child', int), 
                                            ('lambda', float), 
                                            ('weight', float)])

    def compute_stability(self, condensed_tree):
        largest_child = condensed_tree['child'].max()
        smallest_parent = condensed_tree['parent'].min()
        largest_parent = condensed_tree['parent'].max()

        if largest_child < smallest_parent:    # TODO: Why is this necessary? Shouldn't this happen by default?
            largest_child = smallest_parent

        sorted_child_data = np.sort(condensed_tree[['child', 'lambda']], axis=0)
        births = np.nan * np.ones(largest_child + 1)

        # TODO: This is essentially checking whether there are duplicate children and assigning
        #       lambda birth values for children.
        #       Do we need duplicate checking? There should only be a single child instance.
        current_child = -1
        min_lambda = 0

        for child, lambda_val in sorted_child_data: 
            if child == current_child:  # Duplicate child
                min_lambda = min(min_lambda, lambda_val)
                continue
            elif current_child != -1:   # Different child (Assign birth for previous child)
                births[current_child] = min_lambda
            current_child = child       # Initialize and update current child
            min_lambda = lambda_val

        if current_child != -1:
            births[current_child] = min_lambda
        births[smallest_parent] = 0.0
        print("Births:", births)

        parent_keys = np.arange(smallest_parent, largest_parent + 1)
        stability_dict = dict.fromkeys(parent_keys, 0)
        for parent, _, lambda_val, child_size in condensed_tree:    # TODO: I'm worried about how stability scales with singleton clusters
            stability_dict[parent] += (lambda_val - births[parent]) * child_size

        return stability_dict

    def max_lambdas(self, tree):
        sorted_parent_data = np.sort(tree[['parent','lambda']], axis=0)
        deaths = dict()

        current_parent = -1
        max_lambda = 0

        for parent, lambda_val in sorted_parent_data:
            if parent == current_parent:
                max_lambda = max(max_lambda, lambda_val)
                continue
            elif current_parent != -1:
                deaths[current_parent] = max_lambda
            current_parent = parent
            max_lambda = lambda_val
        deaths[current_parent] = max_lambda     # Value for last parent

        return deaths

    def do_labelling(self, tree, clusters):

        allow_single_cluster = True         # TODO: This should probably be a parameter

        root = tree['parent'].min()         # This doubles as the number of points
        labels = np.full(root, -1, dtype=int)
        cluster_label_map = {c: n for n, c in enumerate(clusters)}

        union_find = TreeUnionFind(tree['parent'].max() + 1)    # TODO: Need to either implement or change this
        clusters = set(clusters)
        for parent, child in tree[['parent','child']]:
            if child not in clusters:
                union_find.union(parent, child)

        for n in range(root):
            cluster = union_find.find(n)
            if cluster > root:
                labels[n] = cluster_label_map[cluster]
            elif cluster == root and len(clusters) == 1 and allow_single_cluster:
                if tree['lambda'][tree['child'] == n] >= tree['lambda'][tree['parent'] == cluster].max():
                    labels[n] = cluster_label_map[cluster]

        return labels

    def get_probabilities(self, tree, clusters, labels):
        probabilities = np.zeros(labels.shape[0])
        deaths = self.max_lambdas(tree)
        print("Deaths:", deaths)
        root = tree['parent'].min()

        for _, point, lambda_val, _ in tree[tree['child'] < root]:
            cluster_label = labels[point]
            print("Point:", point, "\tCluster Label:", cluster_label)
            if cluster_label == -1:
                continue

            cluster = clusters[cluster_label]
            max_lambda = deaths[cluster]
            print("Cluster:", cluster, "\tMax Lambda:", max_lambda)
            if max_lambda == 0.0 or not np.isfinite(lambda_val):
                probabilities[point] = 1.0
            else:
                print(min(lambda_val, max_lambda))
                probabilities[point] = min(lambda_val, max_lambda) / max_lambda

        return probabilities

    def get_stability_scores(self, tree, labels, clusters, stability):
        scores = np.empty(len(clusters), dtype=float)
        max_lambda = np.max(tree['lambda'])


        for n, c in enumerate(clusters):
            point_indices = np.where(labels == n)[0]
            cluster_size = np.sum(tree['weight'][np.isin(tree['child'], point_indices)])
            print("Point Indices:", point_indices, "\tCluster Size:", cluster_size)
            if np.isinf(max_lambda) or max_lambda == 0.0 or cluster_size == 0:
                scores[n] = 1.0
            else:
                scores[n] = stability[c] / (cluster_size * max_lambda)
            print("Score:", scores[n])

        return scores

    def get_clusters(self, tree, stability):
        # Assume clusters are ordered by numeric id equivalent to
        # a topological sort of the tree; This is valid given the
        # current implementation above, so don't change that ... or
        # if you do, change this accordingly!

        # TODO: Need to go back through and account for a single root cluster

        max_cluster_size = -1               # TODO: Should this be a parameter? May cause issues with weighty micro-clusters
        allow_single_cluster = True         # TODO: Should this be a parameter?
        cluster_selection_method = 'eom'    # TODO: This should be a parameter if we support leaf clustering

        node_list = sorted(stability.keys(), reverse=True)
        if not allow_single_cluster:
            node_list = node_list[:-1]
        node_list = [int(n) for n in node_list]

        cluster_tree = tree[tree['child'] >= tree['parent'].min()]   # Remove edges that map directly to micro-clusters
        is_cluster = dict.fromkeys(node_list, True) 

        root = tree['parent'].min()

        if max_cluster_size <= 0:
            max_cluster_size = np.inf

        cluster_weights = {child: weight for child, weight in cluster_tree[['child', 'weight']]}
        if allow_single_cluster:
            cluster_weights[root] = np.sum(
                cluster_tree[cluster_tree['parent'] == root]['weight']
            )
            
        if cluster_selection_method == 'eom':
            for node in node_list:  # TODO: Doesn't seem to break in the case of a singleton cluster (ex. 9)
                child_selection = (cluster_tree['parent'] == node)
                subtree_stability = np.sum(
                    [stability[child] for child in cluster_tree['child'][child_selection]]
                )
                if subtree_stability > stability[node] or cluster_weights[node] > max_cluster_size:
                    is_cluster[node] = False
                    stability[node] = subtree_stability
                else:
                    for sub_node in self._bfs_from_cluster_tree(cluster_tree, node):
                        if sub_node != node:
                            is_cluster[sub_node] = False

        elif cluster_selection_method == 'leaf':
            leaves = set(cluster_tree['child'][~np.isin(cluster_tree['child'], cluster_tree['parent'])])
            
            if len(leaves) == 0:                # TODO: Does this account for the case of a single root cluster?
                for c in is_cluster:
                    is_cluster[c] = False
                is_cluster[root] = allow_single_cluster     # TODO: I switched this from always being True, is that right?

            for c in is_cluster:
                is_cluster[c] = (c in leaves)

        else:
            raise ValueError('Invalid Cluster Selection Method: %s\n'
                            'Should be one of: "eom", "leaf"\n')

        clusters = sorted([c for c in is_cluster if is_cluster[c]])
        print(clusters)

        labels = self.do_labelling(tree, clusters)
        print("labels:", labels)
        probs = self.get_probabilities(tree, clusters, labels)
        print("probs:", probs)
        stabilities = self.get_stability_scores(tree, labels, clusters, stability)
        print("stabilities:", stabilities)

        return (labels, probs, stabilities)
