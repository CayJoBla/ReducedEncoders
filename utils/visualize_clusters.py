from matplotlib import pyplot as plt
from matplotlib.colors import Normalize, BoundaryNorm, ListedColormap
import matplotlib.cm as mplcm
import numpy as np
from umap import UMAP

def get_cluster_colormap(num_clusters, cmap='hsv'):
    """Generate a colormap and norm for a given number of clusters.

    Parameters:
        num_clusters (int): The number of clusters.
        cmap (str or matplotlib.colors.Colormap): The base colormap to pull
            colors from. Default is 'hsv'.

    Returns:
        (ListedColormap): The custom colormap for the clusters.
        (BoundaryNorm): The norm for the custom colormap.
    """
    # Get colors of clusters, with grey for outliers
    base_cmap = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    base_norm = Normalize(vmin=0, vmax=num_clusters)
    scalar_map = mplcm.ScalarMappable(norm=base_norm, cmap=base_cmap)
    colors = np.array([scalar_map.to_rgba(i) for i in range(num_clusters)])
    if num_clusters == 0:
        colors = np.array([[0.5, 0.5, 0.5, 1]])
    else:
        colors = np.vstack(([[0.5, 0.5, 0.5, 1]], colors))  # Add gray for outliers

    cluster_cmap = ListedColormap(colors)
    boundaries = np.arange(-1, num_clusters+1)
    cluster_norm = BoundaryNorm(boundaries, cluster_cmap.N, clip=True)

    return cluster_cmap, cluster_norm


def plot_clusters(embeddings, labels, n_components=2, ax=None, cmap="hsv", alpha=0.1, **kwargs):
    """Plot the embeddings colored by cluster. If the embeddings have a higher dimensionality
    than the number of components, they will be reduced to n_components dimensions using UMAP.

    Parameters:
        embeddings (ndarray): The array of embeddings to plot.
        labels (ndarray): The cluster labels for each embedding. -1 is used for noise/outliers.
        n_components (int): The number of dimensions to reduce the embeddings to.
                            Should be either 2 or 3.
        cmap (str or matplotlib.colors.Colormap): The colormap to use.
        alpha (float): The transparency of the points.
    """
    # Reduce the embeddings
    if n_components not in (2,3):
        raise ValueError("n_components must be either 2 or 3.")
    if embeddings.shape[1] < n_components:
        raise ValueError("Embeddings must have at least n_components dimensions.")
    elif embeddings.shape[1] > n_components:
        embeddings = UMAP(n_components=n_components, **kwargs).fit_transform(embeddings)

    # Get number of clusters and check for outliers
    num_labels = len(np.unique(labels))
    if -1 in labels:      # Do not include noise in the number of labels
        num_labels -= 1
        
    # Initialize plot
    if ax is None:  
        fig = plt.figure()
        proj = '3d' if n_components == 3 else 'rectilinear'
        ax = fig.add_subplot(111, projection=proj)

    # Generate colormap
    cmap, norm = get_cluster_colormap(num_labels, cmap=cmap)

    # Plot
    points = ax.scatter(*embeddings.T, alpha=alpha, c=labels, cmap=cmap, norm=norm)
    ax.set_aspect('equal', 'box')

    return points