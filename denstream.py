import copy
import math
from collections import deque
from typing import List
import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.sparse import csr_matrix, csgraph
import networkx as nx
from itertools import combinations

# TODO: What of this algorithm is parallelizable?

class DenStream:
    """My custom implementation of the DenStream algorithm. Adapted from the 
    river library's implementation.

    Though this implementation makes many changes to river's implementation
    to make it more in line with the original paper, it makes the following 
    changes from the paper:
      - The paper's formula for the radius of a micro-cluster often results 
        in negative values, so I've adapted it be non-negative while still
        capturing the idea of standard deviation.
      - The paper makes mention of using a gaussian distribution to 
        approximate a representative point when a micro-cluster is deleted
        that overlaps with another micro-cluster. Though this was already 
        not part of the river implementation, it is also not included in 
        this implementation, due to the high-dimensional nature of our use
        case.
      - In the paper, when p-micro-clusters fall below the threshold weight
        value, they are deleted. In this implementation, an attempt is first 
        made to save that information if possible, either by merging with a 
        nearby micro-cluster or turning it into an o-micro-cluster. This avoids 
        an issue where rising o-micro-clusters could become p-micro-clusters 
        and then be deleted as soon as the timestep advanced and the decay 
        factor was applied.
      - Option for blind initialization, where the algorithm will not perform
        the initial DBSCAN clustering on the first n_samples_init samples. 
        Instead, it just begins with an empty set of micro-clusters, adding 
        points to o-micro-clusters until they become p-micro-clusters. Note 
        that not using blind initialization may add some speedup to the initial
        clustering, but can result in p-micro-clusters that do not satisfy the
        radius condition.
    Some changes that would be nice, but may not be possible:
      - Batched processing of points, to speed up the algorithm. This would
        be especially helpful for our use case, where we are getting points
        from our encoder model in batches as well. Unfortunately, this does 
        not seem very possible under the current model. It may be possible to 
        look into adaptations of the algorithm that allow for batch processing.
    """
    def __init__(
        self,
        decaying_factor: float = 0.25,
        beta: float = 0.75,
        mu: float = 2,
        epsilon: float = 0.02,
        n_samples_init: int = 1000,
        stream_speed: int = 100,
        k: int = 5,
    ):
        super().__init__()
        self.timestamp = 0
        self.decaying_factor = decaying_factor
        self.beta = beta
        self.mu = mu
        self.epsilon = epsilon
        self.n_samples_init = n_samples_init
        self.stream_speed = stream_speed

        self.n_clusters = 0
        self.p_micro_clusters: List[DenStreamMicroCluster] = []
        self.o_micro_clusters: List[DenStreamMicroCluster] = []
        self._core_mask = np.array([], dtype=bool)
        self.clustering = np.array([-1], dtype=int)
        self.is_clustered = False

        # I-HAStream
        self.k = k      # Number of nearest neighbors to consider in core distance calculation
        self._distances = np.array([])
        self.MST = None
        # self._knn_dist = None   # Should be initialized with some initialization step 
        # self._knn_idx = None

        beta_mu = self.beta * self.mu
        self._time_period = math.ceil(
            math.log(beta_mu / (beta_mu - 1)) / self.decaying_factor
        )
        self._n_samples_seen = 0

        # Check that the value of beta is within the range (0,1]
        if not (0 < self.beta <= 1):
            raise ValueError(
                f"The value of `beta` must be within the range (0,1]."
            )
        # Check that the value of mu is greater than 1/beta
        if not (self.mu > 1 / self.beta):
            raise ValueError(
                f"The value of `mu` must be greater than 1/`beta`."
            )

    @property
    def c_micro_clusters(self):
        if not self.is_clustered:   # Recompute which clusters are core clusters
            self._core_mask = np.array(
                [c.calc_weight(self.timestamp) > self.mu for c in self.p_micro_clusters], 
            dtype=bool)
        return np.array(self.p_micro_clusters)[self._core_mask]
    
    def _get_cluster_centers(self, clusters):
        return np.array([c.center for c in clusters], dtype=float)

    def _get_closest_cluster_idx(self, X, clusters, max_dist=np.inf):
        """Get the index of the closest cluster to each point in X. 
        If the closest cluster is further than max_dist, the index will be -1.
        If only one point is passed, a single index will be returned.
        """
        X = np.atleast_2d(X)
        num_points = X.shape[0]
        if len(clusters) == 0:
            return np.full(num_points, -1)
        centers = self._get_cluster_centers(clusters)
        distances = cdist(X, centers)
        idx = np.argmin(distances, axis=-1)
        idx[distances[np.arange(num_points), idx] > max_dist] = -1  # Too far away
        if num_points == 1:    
            idx = idx[0]    # If only one point was passed, return a single index
        return idx

    def _merge(self, x):
        """Merge a new point into the clustering. Either merges it into an existing
        p-micro-cluster, an existing o-micro-cluster, or creates a new o-micro-cluster.
        """
        # Try to merge x into the nearest p-micro-cluster
        if len(self.p_micro_clusters) > 0:
            closest_pmc_idx = self._get_closest_cluster_idx(x, self.p_micro_clusters)
            updated_pmc = copy.deepcopy(self.p_micro_clusters[closest_pmc_idx])
            updated_pmc.insert(x, self.timestamp)
            if updated_pmc.radius <= self.epsilon:
                # Merge the new point into the p-micro-cluster
                self.p_micro_clusters[closest_pmc_idx] = updated_pmc
                return
        
        # Try to merge x into the nearest o-micro-cluster
        if len(self.o_micro_clusters) > 0:
            closest_omc_idx = self._get_closest_cluster_idx(x, self.o_micro_clusters)
            updated_omc = copy.deepcopy(self.o_micro_clusters[closest_omc_idx])
            updated_omc.insert(x, self.timestamp)
            if updated_omc.radius <= self.epsilon:
                # Merge the new point into the o-micro-cluster
                if updated_omc.calc_weight(self.timestamp) > self.mu * self.beta:
                    # The o-micro-cluster becomes a p-micro-cluster
                    del self.o_micro_clusters[closest_omc_idx]
                    self._add_micro_cluster(updated_omc)
                else:
                    self.o_micro_clusters[closest_omc_idx] = updated_omc
                return
        
        # Create a new o-micro-cluster (x was not merged into any existing cluster)
        omc = DenStreamMicroCluster(
            x,
            self.timestamp,
            self.decaying_factor,
        )
        self.o_micro_clusters.append(omc)

    def _adjacency(self, idx):
        """Get the linear indices of the upper triangular adjacency matrix
        for the points adjacent to the given index. Returns a boolean mask
        used to index the _distances array.
        """
        n = len(self.p_micro_clusters)
        i = np.arange(idx)
        j = np.arange(idx+1, n)
        mask = np.zeros(n*(n-1)//2, dtype=bool)
        mask[i*(2*n-i-3)//2 + idx-1] = True
        mask[idx*(2*n-idx-3)//2 + j-1] = True
        return mask

    # def compute_core_dist(self, indices):
    #     adj_dist = np.array([self._distances[self._adjacency(idx)] for idx in indices])
    #     core_dist = np.partition(adj_dist, self.k-1, axis=1)[:,self.k-1]
    #     self.core_distances[indices] = core_dist
    #     return core_dist

    @property
    def core_distances(self):
        return np.partition(squareform(self._distances), self.k-1, axis=1)[:,self.k-1]

    # @property
    # def mreach_distances(self):
    #     i, j = np.triu_indices(len(self.p_micro_clusters), k=1)
    #     dist_triples = np.vstack((self.core_distances[i], self.core_distances[j], self._distances))
    #     return np.max(dist_triples, axis=0)

    @property
    def mreach_distances(self):
        """Returns the matrix of mutual reachability distances between all pairs of
        p-micro-clusters. The mutual reachability distance between two p-micro-clusters
        is the maximum of the core distance of each cluster and the distance between
        the two clusters.
        """
        core_distances = self.core_distances
        core_dist_pairs = np.maximum.outer(core_distances, core_distances)
        return np.clip(core_dist_pairs, squareform(self._distances), None)

    def _insert_to_MST(self, v):
        new_MST = self.MST.copy()
        new_MST.remove_edges_from(new_MST.edges)
        edge_weights = self.mreach_distances
        processed = set()

        def insert_edge(r):     # TODO: ?????
            processed.add(r)
            new_edge = (r, v)
            for w in self.MST.neighbors(r):
                if w not in processed:
                    insert_edge(w)
                    largest_edge = t if edge_weights[t] > edge_weights[new_edge] else new_edge
                    smallest_edge = t if edge_weights[t] < edge_weights[new_edge] else new_edge
                    if l['weight'] < new_edge['weight']:
                        new_edge = l


            t = new_edge['weight']



    def _add_micro_cluster(mc):
        # Determine which core distances need to be updated
        centers = self._get_cluster_centers(self.p_micro_clusters)
        adj_dist = np.linalg.norm(mc.center - centers, axis=-1)
        affected = np.argwhere(adj_dist < self.core_distances)

        # Add micro-cluster and update indexing
        self.p_micro_clusters.append(mc)
        n = len(self.p_micro_clusters)
        self.MST.add_node(n-1)

        # Insert new distances into the array
        adj_mask = self._adjacency(n-1)
        new_distances = np.zeros(n*(n-1)//2)
        new_distances[~adj_mask] = self._distances
        new_distances[adj_mask] = adj_dist
        self._distances = new_distances
        mreach_dist = self.mreach_distances

    def _drop_micro_cluster(idx):
        # Get the distances between micro-cluster to be removed and all other micro-clusters
        adj_mask = self._adjacency(idx)
        adj_dist = self._distances[adj_mask]

        # Remove micro-cluster and update indexing
        mc = self.p_micro_clusters.pop(idx)
        self.MST.remove_node(idx)
        self.MST = nx.convert_node_labels_to_integers(self.MST, ordering='sorted')
        self._distances = self._distances[~adj_mask]

        # Determine which graph nodes need to be updated
        affected = np.argwhere(adj_dist <= self.core_distances) # TODO: In theory, adj_dist should be in the correct order to be compared with core_distances. Check this
        mreach_dist = self.mreach_distances

        # Remove AFFECTED from MST and get remaining supervertices (connected components)
        self.MST.remove_nodes_from(affected)
        supervertices = [list(sv) for sv in nx.connected_components(self.MST)]  # Get supervertices

        # Generate complete graph on supervertices
        supergraph = nx.Graph()
        for sv1, sv2 in combinations(supervertices, 2):
            super_mreach_dist = mreach_dist[np.ix_(sv1, sv2)]
            i, j = np.unravel_index(super_mreach_dist.argmin(), super_mreach_dist.shape)
            best_edge = tuple(sv1[i], sv2[j])
            supergraph.add_edge(sv1, sv2, weight=super_mreach_dist[i,j], edge=best_edge)

        # Compute Prim's MST on supervertices
        super_MST = nx.minimum_spanning_tree(supergraph)

        edges = [(*e['edge'], {'weight':e['weight']}) for _, _, e in super_MST.edges(data=True)]
        self.MST.add_edges_from(edges)

    def _linear_2_triu_idx(k, n):   # TODO: May not need this function
        # Convert 1D array indices into 2D upper triangular indices
        i = n - 2 - np.floor(np.sqrt(-2*k + (n*(n-1) - 1.75)) - 0.5).astype(int)
        j = k + 1 + i*(3 - 2*n + i)//2
        return i, j

    def _get_close_idx_pairs(self, clusters):
        # TODO: This method may achieve a slight speedup with the _linear_2_triu_idx function in the above method
        """Compute the distances between all pairs of cluster centers and return the
        indices of the pairs that are within the sum of the radii of the clusters.
        """
        # Compute pairwise distances between cluster centers
        distances = pdist(self._get_cluster_centers(clusters))

        # Compute the radii of the clusters and pairwise radius sums
        radii = np.array([c.radius for c in clusters])
        radii_sum = radii[:, None] + radii[None, :]
        
        # Find the pairs of clusters that overlap
        i, j = np.triu_indices(len(clusters), k=1)
        linear_indices = np.argwhere(distances <= radii_sum[(i,j)]).ravel()
        
        return i[linear_indices], j[linear_indices]

    def _recluster(self):
        if self.is_clustered:
            return

        c_micro_clusters = self.c_micro_clusters
        num_core_mc = len(c_micro_clusters)
        if num_core_mc == 0:
            self.n_clusters = 0
            self.clustering = np.array([-1], dtype=int)
            self.is_clustered = True
            return

        i, j = self._get_close_idx_pairs(c_micro_clusters)

        G = csr_matrix((np.ones_like(i), (i,j)), shape=(num_core_mc, num_core_mc), dtype=bool)
        n_clusters, labels = csgraph.connected_components(G, directed=False)

        self.n_clusters = n_clusters
        self.clustering = np.append(labels, -1)     # Add a label for outliers
        self.is_clustered = True

        return i, j

    def _recluster_nxgraph(self):
        if self.is_clustered:
            return

        c_micro_clusters = self.c_micro_clusters
        num_core_mc = len(c_micro_clusters)
        if num_core_mc == 0:
            self.n_clusters = 0
            self.clustering = np.array([-1], dtype=int)
            self.is_clustered = True
            return

        i, j = self._get_close_idx_pairs(c_micro_clusters)
        
        G = nx.Graph()
        G.add_nodes_from(range(num_core_mc))
        G.add_edges_from(zip(i, j))

        # clusters = []
        # for component in nx.connected_components(G):#, backend='cugraph'):
        #     component = np.fromiter(component, int, len(component))
        #     cluster_elements = copy.deepcopy(c_micro_clusters[component])
        #     cluster = cluster_elements[0]
        #     for c in cluster_elements[1:]:
        #         cluster.merge(c)
        #     clusters.append(cluster)
        # self.n_clusters, self.clusters = len(clusters), clusters

        labels = np.full(num_core_mc, -1)
        for i, component in enumerate(nx.connected_components(G)):
            labels[list(component)] = i
            print(component)

        self.n_clusters = i + 1
        self.clustering = np.append(labels, -1)     # Add a label for outliers
        self.is_clustered = True

    def _prune(self):
        self.p_micro_clusters = [
            pmc for pmc in self.p_micro_clusters 
            if pmc.calc_weight(self.timestamp) >= self.beta * self.mu
        ]

        f = lambda t: 2**(-self.decaying_factor * t)
        xi = lambda omc: (f(self.timestamp - omc.creation_time + self._time_period) - 1) / (f(self._time_period) - 1)
        self.o_micro_clusters = [
            omc for omc in self.o_micro_clusters 
            if omc.calc_weight(self.timestamp) >= xi(omc)
        ]

    def _prune_safe(self):
        # Get the indices of the p-micro-clusters that need to be pruned
        pmc_prune_indices = [
            i for i, pmc in enumerate(self.p_micro_clusters)
            if pmc.calc_weight(self.timestamp) < self.beta * self.mu
        ]
        for i in reversed(pmc_prune_indices):
            # For each p-micro-cluster to be pruned, try to save that information
            pmc = self.p_micro_clusters.pop(i)
            x = pmc.center
            
            # Try to merge with the nearest p-micro-cluster
            if len(self.p_micro_clusters) > 0:
                closest_pmc_idx = self._get_closest_cluster_idx(x, self.p_micro_clusters)
                updated_pmc = copy.deepcopy(self.p_micro_clusters[closest_pmc_idx])
                updated_pmc.merge(pmc)
                is_valid_pmc = updated_pmc.calc_weight(self.timestamp) >= self.beta*self.mu
                if is_valid_pmc and updated_pmc.radius <= self.epsilon:
                    # Merge the two p-micro-clusters
                    self.p_micro_clusters[closest_pmc_idx] = updated_pmc
                    continue
            
            # Try to merge into the nearest o-micro-cluster
            if len(self.o_micro_clusters) > 0:
                closest_omc_idx = self._get_closest_cluster_idx(x, self.o_micro_clusters)
                updated_omc = copy.deepcopy(self.o_micro_clusters[closest_omc_idx])
                updated_omc.merge(pmc)
                if updated_omc.radius <= self.epsilon:
                    # Merge the decayed p-micro-cluster into the o-micro-cluster
                    if updated_omc.calc_weight(self.timestamp) > self.mu * self.beta:
                        # The o-micro-cluster becomes a p-micro-cluster
                        del self.o_micro_clusters[closest_omc_idx]
                        self.p_micro_clusters.append(updated_omc)
                    else:
                        self.o_micro_clusters[closest_omc_idx] = updated_omc
                    continue
            
            # Turn the decayed p-micro-cluster into an o-micro-cluster
            self.o_micro_clusters.append(pmc)

        # Prune the o-micro-clusters that are least likely to develop into p-micro-clusters
        f = lambda t: 2**(-self.decaying_factor * t)
        xi = lambda omc: (f(self.timestamp - omc.creation_time + self._time_period) - 1) / (f(self._time_period) - 1)
        self.o_micro_clusters = [
            omc for omc in self.o_micro_clusters 
            if omc.calc_weight(self.timestamp) >= xi(omc)
        ]

    def learn_one(self, x):
        self.is_clustered = False
        self._n_samples_seen += 1
        if self._n_samples_seen % self.stream_speed == 0:
            self.timestamp += 1

        # Merge
        self._merge(x)
            
        if self.timestamp > 0 and self.timestamp % self._time_period == 0:
            self._prune()

    def predict_one(self, x):
        return self.predict(np.atleast_2d(x))

    def predict(self, X):
        # Initialization
        if (self.MST is None) and (self._n_samples_seen >= self.n_samples_init) and \
                (len(self.p_micro_clusters) >= self.min_clusters_init):
            self._distances = pdist(self._get_cluster_centers(self.p_micro_clusters))
            self.MST = nx.minimum_spanning_tree(nx.Graph(self.mreach_distances))

        self._recluster()   # TODO: Make this work for the new setup
        closest_idx = self._get_closest_cluster_idx(X, self.c_micro_clusters, max_dist=self.epsilon)
        return self.clustering[closest_idx]


class DenStreamMicroCluster:
    """DenStream Micro-cluster class"""

    def __init__(self, X, timestamp, decaying_factor):
        """Initialize a new micro-cluster. 
        X can be a single point or a 2d array of points.
        """
        self.last_edit_time = timestamp
        self.creation_time = timestamp
        self.decaying_factor = decaying_factor

        X = np.atleast_2d(X)
        self._cf1 = np.sum(X, axis=0, dtype=float)
        self._cf2 = np.sum(np.square(X), axis=0, dtype=float)
        self._w = float(X.shape[0])

    def fading_function(self, t):
        return 2 ** (-self.decaying_factor * t)

    def _update(self, timestamp):
        """Updates the running statistics of the micro-cluster by applying the
        fading function according to the time elapsed since the last update.
        """
        if self.last_edit_time == timestamp:
            return
        fading_factor = self.fading_function(timestamp - self.last_edit_time)
        self._cf1 *= fading_factor 
        self._cf2 *= fading_factor
        self._w *= fading_factor
        self.last_edit_time = timestamp

    def calc_weight(self, timestamp):
        """Computes the weight of the micro-cluster."""
        self._update(timestamp)
        return self._w

    @ property
    def center(self):
        """Computes the center of the micro-cluster."""
        return self._cf1 / self._w

    @property
    def radius(self):
        """Computes the radius of the micro-cluster.
        Note that this forumla for the radius does not match the formula
        presented in the original paper, due to the fact that the original
        formula resulted in negative values for the radius. 
        The formula used here was derived from the formula: 
            std(X) = sqrt(var(X)) = sqrt(E[X^2] - E[X]^2)
        and is given by: 
            r = || sqrt( (CF2 / w) - (CF1 / w)^2 ) ||
        In this case, the standard deviation is computed in each dimension
        and the norm of the resulting vector is taken to get the radius.
        """
        variances = (self._cf2 / self._w) - np.square(self._cf1 / self._w)
        return np.sqrt(np.sum(variances))
        
    def insert(self, X, timestamp):
        """Inserts a new point or array of points into the micro-cluster."""
        self._update(timestamp)
        X = np.atleast_2d(X)
        self._cf1 += np.sum(X, axis=0)
        self._cf2 += np.sum(np.square(X), axis=0)
        self._w += X.shape[0]
        return self

    def merge(self, other):
        """Merges another micro-cluster into this micro-cluster"""
        timestamp = max(self.last_edit_time, other.last_edit_time)
        self._update(timestamp)
        other._update(timestamp)
        self._cf1 += other._cf1
        self._cf2 += other._cf2
        self._w += other._w
        return self
