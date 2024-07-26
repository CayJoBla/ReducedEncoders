# <center> HDBSCAN </center>

For most of this document, I looked at the `hdbscan` documentation [here](https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html) to get an understanding of the algorithm. I go over the basic steps of the algorithm as well as how I think it can be used in our research context. The main questions I am looking to answer are:
- How does HDBSCAN work?
- How does adding new points affect an existing HDBSCAN clustering?
- Can adding new points be done without re-running the entire algorithm?
- What potential issues are there when adding new points?

## The Basics
HDBSCAN is a hierarchical, density-based clustering algorithm. This means that the algorithm creates a hierarchy of clusters, where each cluster is defined in some way by the density of the points within it. Different from centroid-based clustering algorithms, it does not make assumptions about the shapes of the clusters.

The two main parameters for the HDBSCAN algorithm are:
- `k`, the number of nearest neighbors to use in distance computations, and 
- `min_cluster_size`, the minimum number of points required to form a cluster, which helps determine which points are considered noise.

## The Algorithm
### Main Ideas
Core Distance
: The distance to the `k`-th nearest neighbor of a point. This is a measure of how dense the area around the point is. Can also be thought of as the radius around a point needed for the resulting area to contain `k` neighbors.

Mutual Reachability Distance
: A distance metric used to "spread apart points with low density". We do this by forcing the smallest distance between any two points to be the maximum of their core distances, or $$d_{\text{mreach-}k}(a,b) := \max{\{\text{core}_k(a), \text{core}_k(a), d(a,b)\} }$$ where $d(a,b)$ is the (euclidean) distance between points $a$ and $b$.

Cluster Stability
: A value used to determine whether to split a cluster, given by $$\sum_{p \in \text{cluster}} (\lambda_p - \lambda_{\text{birth}})$$ Note that $\lambda = \frac{1}{\text{distance}}$, so $\lambda_{\text{birth}}$ is the value of $\lambda$ when the cluster was split from its parent, and $\lambda_p$ is the value of $\lambda$ when point $p$ was dropped from the cluster.

***Minimum Spanning Trees***
***Prim's Algorithm / Boruvka's Algorithm***
***Condensed Trees***

### Steps
1. **Compute Core Distances**: For each point, compute the core distance.
2. **Compute Mutual Reachability Distance**: For each pair of points, compute the mutual reachability distance.
3. **Build Minimum Spanning Tree**: Use the mutual reachability distances to build a minimum spanning tree using Prim's algorithm or Boruvka's algorithm.
4. **Create a Cluster Hierarchy**: Sort the edges of the minimum spanning tree by increasing mutual reachability distance to create a cluster hierarchy.
5. **Create the Condensed Tree**: For each cluster split in the hierarchy, determine if the resulting clusters have the `min_cluster_size` number of points. For the resulting clusters that do not, those points are considered noise, and are dropped from the parent cluster. If both resulting clusters have the minimum number of points, the cluster is split into two new clusters. This results in a condensed tree.
6. **Extract Clusters**: Using the condensed tree, the cluster stability is computed for each cluster. If the parent cluster has a higher cluster stability than the sum of the stabilities of the child clusters, the parent cluster is kept. Otherwise, the parent splits and the child clusters are kept. This results in the final clusters.

## Adding New Points / Potential Issues
Let us consider the case of adding a single new point to our existing clustering. Because of the way that HDBSCAN is done, it seems that online clustering with new data is not possible. With the addition of a new point, the core distances and mutual reachability distances of some subset if not all points would need to be recomputed, as well as the minimum spanning tree, cluster hierarchy, condensed tree, and final clusters. Perhaps some shortcuts can be made to reduce the computation time, such as looking into a more efficient way to determine what distances need to be recomputed, or updating the minimum spanning tree from the existing one, but the algorithm would still need to pretty much be run from the beginning as I understand it.

The next best route then seems to be to just run predictions on the new data with the existing clustering. Then, at some point, the entire algorithm can be run again with the new data included, including some efficiency speedups if possible. The question then becomes: "When should the clustering algorithm be re-run?"

### Ideas
Some ideas for when to re-run the clustering algorithm.
- **Time-based**: Run the algorithm every $t$ minutes. Can also consider how long the algorithm takes to run and run it when the system is not busy.
- **Data-based**: Run the algorithm when the number of new points reaches a certain threshold. This could be a fixed number or a percentage of the existing data.
- **Cluster Stability-based**: Since clusters are split based on cluster stability in the HDBSCAN algorithm, we should be able to compute or approximate the stability of the existing clusters with new data and re-run the algorithm when the stability indicates that a cluster should be split. Note that this does not account for new clusters being formed from noise points, though it could perhaps be combined with some checking for minimum cluster size on the minimum spanning tree.

### Cluster Stability-based Update Method
Since time-based and data-based methods are relatively straightforward, I looked into the viability of the cluster stability-based update method. Unfortunately, because the code is not well-documented, and their function that I hoped to use is both riddled with bugs and is not used anywhere else in the repo, even following along with their implementation of HDBSCAN led to errors that I could not resolve. Furthermore, I determined that if the cluster that should be split is a leaf cluster in the condensed tree, the algorithm would not have children nodes to check the stability of to determine when to split. There is also an issue of clusters that emerge from noise points. Since each new point could dramatically change the core distances and mutual reachability distances of the existing points, which would then change the MST and condensed tree, this case could not be handled by the cluster stability-based method.

I believe there are other methods that could be used to assess the stability of our existing clusters, which we could use to check our clusters each time that new points are added. We would need to choose an assessment method that also looks at noise points if we want to solve that issue as well. As it stands, it is difficult for me to come up with a more clever way to determine when to re-run the algorithm.

