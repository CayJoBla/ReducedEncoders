import copy
import math
from collections import deque
from typing import List
import numpy as np
from scipy.spatial.distance import cdist, pdist
from scipy.sparse import csr_matrix, csgraph
import networkx as nx
from scipy.sparse import csr_matrix
from bertopic.vectorizer import ClassTfidfTransformer

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
        term_threshold: float = 0.1,
    ):
        super().__init__()
        self.timestamp = 0
        self.decaying_factor = decaying_factor
        self.beta = beta
        self.mu = mu
        self.epsilon = epsilon
        self.n_samples_init = n_samples_init
        self.stream_speed = stream_speed
        self.term_threshold = term_threshold    # Weight threshold to drop terms from micro-clusters

        self.n_clusters = 0
        self.p_micro_clusters: List[DenStreamMicroCluster] = []
        self.o_micro_clusters: List[DenStreamMicroCluster] = []
        self._core_mask = np.array([], dtype=bool)
        self.clustering = np.array([-1], dtype=int)
        self.is_clustered = False

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

    # def _get_closest_cluster_idx(self, point, clusters, max_dist=math.inf):
    #     centers = self._get_cluster_centers(clusters)
    #     distances = self._distance(centers, point)
    #     idx = np.argmin(distances)
    #     return idx if distances[idx] <= max_dist else -1

    def _merge(self, x, document=None):
        """Merge a new point into the clustering. Either merges it into an existing
        p-micro-cluster, an existing o-micro-cluster, or creates a new o-micro-cluster.
        """
        # Try to merge x into the nearest p-micro-cluster
        if len(self.p_micro_clusters) > 0:
            closest_pmc_idx = self._get_closest_cluster_idx(x, self.p_micro_clusters)
            updated_pmc = copy.deepcopy(self.p_micro_clusters[closest_pmc_idx])
            updated_pmc.insert(x, document, self.timestamp)
            if updated_pmc.radius <= self.epsilon:
                # Merge the new point into the p-micro-cluster
                self.p_micro_clusters[closest_pmc_idx] = updated_pmc
                self.is_clustered = False   # TODO: Ideally, we instead want to set a threshold for how much changes before we need to recluster
                return
        
        # Try to merge x into the nearest o-micro-cluster
        if len(self.o_micro_clusters) > 0:
            closest_omc_idx = self._get_closest_cluster_idx(x, self.o_micro_clusters)
            updated_omc = copy.deepcopy(self.o_micro_clusters[closest_omc_idx])
            updated_omc.insert(x, document, self.timestamp)
            if updated_omc.radius <= self.epsilon:
                # Merge the new point into the o-micro-cluster
                if updated_omc.calc_weight(self.timestamp) > self.mu * self.beta:
                    # The o-micro-cluster becomes a p-micro-cluster
                    del self.o_micro_clusters[closest_omc_idx]
                    self.p_micro_clusters.append(updated_omc)
                    self.is_clustered = False   # New p-mc, we need to recluster
                else:
                    self.o_micro_clusters[closest_omc_idx] = updated_omc
                return
        
        # Create a new o-micro-cluster (x was not merged into any existing cluster)
        omc = DenStreamMicroCluster(
            x,
            document,
            self.timestamp,
            self.decaying_factor,
            self.term_threshold,
        )
        self.o_micro_clusters.append(omc)

    # def _is_directly_density_reachable(self, c_p, c_q):
    #     """Check if two clusters are directly density reachable."""
    #     # Both clusters are core micro-clusters
    #     if self._is_core_cluster(c_p) and self._is_core_cluster(c_q):
    #         distance = self._distance(c_p.center, c_q.center)
    #         if distance < 2 * self.epsilon:             # Clusters are close
    #             if distance <= c_p.radius + c_q.radius: # Clusters are overlapping
    #                 return True
    #     return False

    # def _is_core_cluster(self, cluster):
    #     return cluster.calc_weight(self.timestamp) > self.mu

    # def _query_neighbor(self, cluster):
    #     neighbors = []
    #     if self._is_core_cluster(cluster):
    #         for pmc in self.p_micro_clusters:
    #             if cluster != pmc and self._is_directly_density_reachable(cluster, pmc):
    #                 neighbors.append(pmc)
    #     return neighbors

    # @staticmethod
    # def _generate_clusters_for_labels(cluster_labels):
    #     clusters = {}
    #     for mc, label in cluster_labels.items():
    #         cluster = copy.deepcopy(mc)     # Don't modify the original micro-cluster
    #         if label not in clusters:
    #             clusters[label] = cluster
    #         else:
    #             clusters[label].merge(cluster)
    #         # merge clusters with the same label into a big cluster
    #         cluster = copy.deepcopy(micro_clusters[0])
    #         for mc in range(1, len(micro_clusters)):
    #             cluster.merge(micro_clusters[mc])

    #         clusters[label] = cluster

    #     return len(clusters), clusters

    # def _get_neighborhood_ids(self, item):
    #     neighborhood_ids = deque()
    #     for idx, other in enumerate(self._init_buffer):
    #         if not other.covered:
    #             if self._distance(item.x, other.x) < self.epsilon:
    #                 neighborhood_ids.append(idx)
    #     return neighborhood_ids

    def _linear_2_triu_idx(k, n):   # TODO: May not need this function
        # Convert 1D array indices into 2D upper triangular indices
        i = n - 2 - np.floor(np.sqrt(-2*k + (n*(n-1) - 1.75)) - 0.5).astype(int)
        j = k + 1 + i*(3 - 2*n + i)//2
        return i, j

    # def _initial_dbscan(self):
    #     # Find the core points and their neighbors
    #     X = np.array(self._init_buffer)
    #     distances = cdist(X, X)
    #     connected = distances <= self.epsilon
    #     is_core_point = np.sum(connected, axis=1) >= (self.beta * self.mu)
    #     np.fill_diagonal(connected, False)  # Don't count the point itself as a neighbor
    #     i_close, j_close = np.argwhere(connected).T

    #     get_neighbors = lambda i: deque(j_close[i_close == i])
       
    #     # Initialize values for the BFS
    #     visited = set()
    #     p_micro_clusters = []

    #     # Iterate over the core points
    #     for i in np.arange(X.shape[0])[is_core_point]:
    #         if i in visited:    # Skip if visited
    #             continue

    #         visited.add(i)      # Mark as visited
    #         cluster = [i]
    #         neighbors = get_neighbors(i)
    #         while neighbors:
    #             j = neighbors.popleft()
    #             if j in visited:        # If in a different cluster, skip
    #                 continue            # (already a border point of another cluster)
    #             visited.add(j)
    #             cluster.append(j)
    #             if is_core_point[j]:    # If j is a core point, add its neighbors
    #                 neighbors += get_neighbors(j)
    #         p_micro_clusters.append(    # Create a new p-micro-cluster
    #             DenStreamMicroCluster(
    #                 X[cluster], 
    #                 self.timestamp, 
    #                 self.decaying_factor)
    #         )
    #     self.p_micro_clusters = p_micro_clusters

    # TODO: Need to do testing on which of these two versions is faster
    #       Tests and logic indicate that the second version is faster, since the clusters
    #       should already be updated by the check on core clusters.  

    # def _get_close_idx_pairs(self, clusters):
    #     """Compute the distances between all pairs of cluster centers and return the
    #     indices of the pairs that are within 2*epsilon of each other.
    #     """
    #     # Compute pairwise distances between cluster centers
    #     n_clusters = clusters.shape[0]
    #     distances = pdist(self._get_cluster_centers(clusters))
    #     linear_indices = np.argwhere(distances < 2*epsilon).ravel() # Find the close pairs
    #     i, j = self._linear_2_triu_idx(linear_indices, n_clusters)  # Get index pairs

    #     # Of the close pairs, find those that are overlapping
    #     close_cluster_indices = np.unique((i, j))
    #     radii = np.array([clusters[i].radius if i in close_cluster_indices
    #                       else 0 for i in range(n_clusters)])
    #     pair_indices = np.argwhere(distances[linear_indices] < radii[i]+radii[j]).ravel()
    #     return i[pair_indices], j[pair_indices]

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

    # TODO: Need to test which of the recluster methods is fastest (or is the original method fastest?)
    # def _recluster(self):
    #     # This function handles the case when a clustering must be recomputed.
    #     if self.is_clustered:
    #         return      # No need to recompute clusters 

    #     c_micro_clusters = self.c_micro_clusters
    #     if len(c_micro_clusters) == 0:
    #         self.clusters = []
    #         self.n_clusters = 0
    #         self.is_clustered = True
    #         return
    #     i_close, j_close = self._get_close_idx_pairs(c_micro_clusters)

    #     # Define a function to get the undirected neighbors of a cluster
    #     get_neighbors = lambda i: deque(j_close[i_close == i]) + deque(i_close[j_close == i])

    #     visited = set()
    #     clusters = []
    #     for i in range(len(c_micro_clusters)):
    #         if i in visited:
    #             continue

    #         # Start a new cluster
    #         clusters.append(copy.deepcopy(c_micro_clusters[i]))
    #         visited.add(i)

    #         neighbors = get_neighbors(i)
    #         while neighbors:    # BFS to find connected components
    #             j = neighbors.popleft()
    #             if j not in visited:
    #                 clusters[-1].merge(copy.deepcopy(c_micro_clusters[j]))
    #                 visited.add(j)
    #                 neighbors += get_neighbors(j)   # Add neighbors of j to the queue

    #     self.n_clusters, self.clusters = len(clusters), clusters
    #     self.is_clustered = True

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

    # def _recluster(self):
    #     # This function handles the case when a clustering must be recomputed.
    #     if self.is_clustered:
    #         return      # No need to recompute clusters 

    #     # cluster counter; in this algorithm cluster labels start with 0
    #     c = -1
    #     # initiate labels of p-micro-clusters to None
    #     labels = {pmc: None for pmc in self.p_micro_clusters.values()}

    #     for pmc in self.p_micro_clusters.values():
    #         # previously processed in inner loop
    #         if labels[pmc] is not None:
    #             continue
    #         # next cluster label
    #         c += 1
    #         labels[pmc] = c
    #         # neighbors to expand
    #         seed_queue = self._query_neighbor(pmc)
    #         # process every point in seed set
    #         while seed_queue:
    #             # check previously proceeded points
    #             if labels[seed_queue[0]] is not None:
    #                 seed_queue.popleft()
    #                 continue
    #             if seed_queue:
    #                 labels[seed_queue[0]] = c
    #                 # find neighbors of neighbors
    #                 neighbor_neighbors = self._query_neighbor(seed_queue[0])
    #                 # add new neighbors to seed set
    #                 for neighbor_neighbor in neighbor_neighbors:
    #                     if labels[neighbor_neighbor] is None:
    #                         seed_queue.append(neighbor_neighbor)

    #     self.n_clusters, self.clusters = self._generate_clusters_for_labels(labels)
    #     self.is_clustered = True

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

    def learn_one(self, x, document=None):
        self._n_samples_seen += 1
        if self._n_samples_seen % self.stream_speed == 0:
            self.timestamp += 1

        # # Initialization
        # if not self.initialized:
        #     self._init_buffer.append(x)
        #     if self._n_samples_seen >= self.n_samples_init:
        #         self._initial_dbscan()
        #         self.initialized = True
        #         del self._init_buffer
        #     return

        # Merge
        self._merge(x, document)

        if self.timestamp > 0 and self.timestamp % self._time_period == 0:
            self._prune()

    def predict_one(self, x):
        return self.predict(np.atleast_2d(x))

    def predict(self, X):
        self._recluster()
        closest_idx = self._get_closest_cluster_idx(X, self.c_micro_clusters, max_dist=self.epsilon)
        return self.clustering[closest_idx]

    def tf_query(self):
        """Build a sparse weight matrix of the terms in each cluster.
        This matrix can be used to get c-TF-IDF values for each cluster.
        """
        self._recluster()   # Ensure that the clusters are up-to-date

        terms = set()       # Get all terms
        for i, mc in enumerate(self.p_micro_clusters):
            terms.update(mc.tf.keys())
        terms = sorted(terms)

        # Build the sparse matrix
        data = csr_matrix((self.n_clusters, len(terms)), dtype=float)
        for i, label in enumerate(self.clustering):
            mc = self.p_micro_clusters[i]
            for j, term in enumerate(terms):
                if term in mc.tf:
                    data[label, j] += mc.tf[term]

        return data, terms

    def c_tf_idf(self):
        data, terms = self.tf_query()
        transformer = ClassTfidfTransformer()
        return transformer.fit_transform(data).toarray(), terms


class DenStreamMicroCluster:
    """DenStream Micro-cluster class"""

    def __init__(self, X, documents, timestamp, decaying_factor, term_threshold=0.1):
        """Initialize a new micro-cluster. 
        X can be a single point or a 2d array of points.
        """
        # Time values
        self.last_edit_time = timestamp
        self.creation_time = timestamp
        self.decaying_factor = decaying_factor

        # Spatial values
        X = np.atleast_2d(X)
        self._cf1 = np.sum(X, axis=0, dtype=float)
        self._cf2 = np.sum(np.square(X), axis=0, dtype=float)
        self._w = float(X.shape[0])

        # c-TF-IDF values
        self.term_threshold = term_threshold
        if documents is not None:
            documents = np.ravel(documents)
            self._term_w = float(len(documents))                            # Weighted number of terms in micro-cluster
            words, counts = np.unique(documents, return_counts=True)
            self.tf = {word: count for word, count in zip(words, counts)}   # Weighted term frequency
        else:
            self._term_w = 0
            self.tf = {}

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
        self._term_w *= fading_factor
        dropped_terms = []
        for k, v in self.tf.items():
            self.tf[k] = v * fading_factor
            if self.tf[k] < self.term_threshold:
                dropped_terms.append(k)
        for k in dropped_terms:
            del self.tf[k]
            
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
        
    def insert(self, x, document, timestamp):
        """Inserts a single point into the micro-cluster."""
        self._update(timestamp)

        self._cf1 += x
        self._cf2 += np.square(x)
        self._w += 1

        if document is not None:
            document = np.ravel(document)
            self._term_w += len(document)
            words, counts = np.unique(document, return_counts=True)
            for word, count in zip(words, counts):
                self.tf[word] = self.tf.get(word, 0) + count

        return self

    def merge(self, other):
        """Merges another micro-cluster into this micro-cluster"""
        timestamp = max(self.last_edit_time, other.last_edit_time)
        self._update(timestamp)
        other._update(timestamp)

        self._cf1 += other._cf1
        self._cf2 += other._cf2
        self._w += other._w

        self._term_w += other._term_w
        for word, count in other.tf.items():
            self.tf[word] = self.tf.get(word, 0) + count

        return self
