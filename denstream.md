# <center> DenStream </center>

My reference for this document is [this paper](https://www.researchgate.net/publication/220906738_Density-Based_Clustering_over_an_Evolving_Data_Stream_with_Noise), which introduces the algorithm. Furthermore, I look at the [documentation](https://riverml.xyz/dev/api/cluster/DenStream/) of the River implementation. 

The purpose of this document is to summarize the DenStream algorithm and to discuss how it can be used in our research context. The main questions I am looking to answer are:
- How does DenStream work?
- How does adding new points affect an existing DenStream clustering (i.e. are the new clusters drastically different from the old ones, or are updates fairly consistent with older data, so we can see how clusters drift over time)?
- Does the algorithm allow for cluster splitting/merging, or just emergence/disappearance of clusters?
- What limitations does DenStream have with respect to our research context (i.e. spatial complexity, difficulty in high-dimensional data, etc.)?
- How does DenStream compare to other clustering algorithms?

## Overview
DenStream is a clustering algorithm that is designed to work with data streams. It is able to discover clusters with arbitrary shapes and sizes, and is able to handle noise points. The algorithm is based on the idea of microclusters, which are small clusters that are created from the data stream. These microclusters are then used to create larger clusters, which are the final clusters that are output by the algorithm.

The DenStream algorithm makes use of multiple parameters, including:
- `lambda`: The decaying factor. Determines the importance of historical data to current cluster. The higher the value of $\lambda$, the lower importance of the historical data compared to more recent data.
- `mu`: Parameter to determine the minimum weight of a micro-cluster to be considered a core micro-cluster. Since $\beta \mu > 1$, $\mu \in (\frac{1}{\beta}, \infty)$.
- `beta`: Parameter to determine the fraction of the weight limit, $\mu$, that a micro-cluster must have to be considered a potential core micro-cluster. $\beta \in (0,1)$.
- `epsilon`: Defines the neighborhood around a micro-cluster. The radius of any micro-cluster must be less than or equal to $\epsilon$.
- `n_samples_init`: The number of samples to use for initializing the algorithm and the micro-clusters.
- `v`: The rate of the data stream, or the number of data points per time unit.

## The Algorithm
The DenStream algorithm is divided into two main parts: online and offline. The online part is responsible for maintaining the micro-clusters as new data points arrive, while the offline part is responsible for extracting the final clusters from the micro-clusters when the user requests them.

### Definitions
Core-Micro-Cluster (`c-micro-cluster`)
:  A group of close points $p_1$,$p_2$,...,$p_n$ with timestamps $T_1$,$T_2$,...,$T_n$ such that $w \geq \mu$ and $r \leq \epsilon$. Given $f(t) = 2^{-\lambda t}$, the CMC at time $t$ is defined by the following parameters:
    - weight $w := \sum_{i=1}^n f(t-T_i)$ 
    - center $c := \frac{1}{w}\sum_{i=1}^n f(t-T_i)p_i$
    - radius $r := \frac{1}{w}\sum_{i=1}^n f(t-T_i)d(p_i, c)$

Potential Core-Micro-Cluster (`p-micro-cluster`)
: A group of close points $p_1$,$p_2$,...,$p_n$ with timestamps $T_1$,$T_2$,...,$T_n$ such that $w \geq \beta \mu$ and $r \leq \epsilon$. Defined by $\{\overline{CF^1}, \overline{CF^2}, w\}$ with the following parameters:
    - $\overline{CF^1} := \sum_{i=1}^n f(t-T_i) p_i$ 
    - $\overline{CF^2} := \sum_{i=1}^n f(t-T_i) p_i^2$ 
    - weight $w := \sum_{i=1}^n f(t-T_i)$ 
    - center $c := \frac{\overline{CF^1}}{w}$
    - radius $r := \sqrt{\frac{\lvert\overline{CF^2}\rvert}{w} - c^2}$

Outlier Micro-Cluster (`o-micro-cluster`)
: A group of close points $p_1$,$p_2$,...,$p_n$ with timestamps $T_1$,$T_2$,...,$T_n$ such that $w \geq \beta \mu$ and $r \leq \epsilon$. Defined by $\{\overline{CF^1}, \overline{CF^2}, w, t_o\}$ as in the p-micro-cluster, but with the additional parameter $t_o = T_1$ denoting the creation time of the o-micro-cluster.

Directly Density-Reachable
: A p-micro-cluster $c_p$ is directly density-reachable from a p-micro-cluster $c_q$ wrt. $\epsilon$ and $\mu$ if the weight of $c_q$ is above $\mu$ (i.e., $c_q$ corresponds a c-micro-cluster) and $d(c_p, c_q) \leq 2\epsilon$, where $d(c_p, c_q)$ is the distance between the centers of $c_p$ and $c_q$.

Density-Reachable
: A p-micro-cluster $c_p$ is density-reachable from a p-micro-cluster $c_q$ wrt. $\epsilon$ and $\mu$ if there is a chain of p-micro-clusters $c_{p_1}$, $c_{p_2}$, ..., $c_{p_n}$ such that $c_{p_1} = c_q$, $c_{p_n} = c_p$, and $c_{p_{i+1}}$ is directly density-reachable from $c_{p_i}$.

Density-Connected
: A p-micro-cluster $c_p$ is density-connected to a p-micro-cluster $c_q$ wrt. $\epsilon$ and $\mu$ if there is a p-micro-cluster $c_m$ such that both $c_p$ and $c_q$ are density-reachable from $c_m$ wrt. $\epsilon$ and $\mu$.

### Online (Micro-Cluster Maintenance)

#### Merging($p$, $\epsilon$, $\beta$, $\mu$)
Merge a point $p$ into the closest p-micro-cluster or o-micro-cluster. If the point is not within $\epsilon$ of any existing micro-cluster, create a new o-micro-cluster.
1. **Try merging into a p-micro-cluster**
    - Find the closest p-micro-cluster $c_p$ to $p$
    - With $p$ in $c_p$, if new radius $r_p \leq \epsilon$ then 
        - Merge $p$ into $c_p$
2. **Else: Try merging into an o-micro-cluster**
    - Find the closest o-micro-cluster $c_o$ to $p$
    - With $p$ in $c_o$, if new radius $r_o \leq \epsilon$ then 
        - Merge $p$ into $c_o$
        - If $w_o > \beta \mu$ then 
            - Convert $c_o$ to a p-micro-cluster
3. **Else: Create a new o-micro-cluster**
    - Create a new o-micro-cluster $c_o$ with $p$ as the only point

#### Pruning($t$, $\beta$, $\mu$)
Check the weights of all p-micro-clusters and o-micro-clusters to see if they have decayed. If the weight of a p-micro-cluster is less than $\beta\mu$, remove it. If the weight of an o-micro-cluster is less than the lower weight limit, remove it. In general, this is run every $T_p$ time units, where 
$$T_p = \left\lceil \frac{1}{\lambda} \log{\left(\frac{\beta\mu}{\beta\mu-1}\right)} \right\rceil.$$
1. **Check p-micro-clusters**
    - For each p-micro-cluster $c_p$:
        - If $w_p < \beta\mu$ then 
            - Delete $c_p$
2. **Check o-micro-clusters**
    - For each o-micro-cluster $c_o$:
        - $\xi = \frac{2^{-\lambda(t-t_o+T_p)}-1}{2^{-\lambda T_p}-1}$
        - If $w_o < \xi$ then 
            - Delete $c_o$

### Offline (Cluster Extraction)

#### DBSCAN($\epsilon$, $\mu$)
Run a variant of DBSCAN on the p-micro-clusters to extract the final clusters. All the density-connected p-micro-clusters are grouped together to form a cluster.


## Limitations / Potential Issues
