from matplotlib import pyplot as plt
from matplotlib import colors
import matplotlib.cm as mplcm
import numpy as np
from umap import UMAP

def plot_clusters(embeddings, labels, n_components=2, ax=None, cmap="hsv", alpha=0.1, **kwargs):
    """Reduce the embeddings and plot them colored by cluster.

    Parameters:
        embeddings (ndarray): The array of embeddings to plot.
        labels (ndarray): The cluster labels for each embedding. -1 is used for noise.
        n_components (int): The number of dimensions to reduce the embeddings to.
                            Should be either 2 or 3.
        cmap (str or matplotlib.colors.Colormap): The colormap to use.
        alpha (float): The transparency of the points.
    """
    # Reduce the embeddings
    if n_components not in (2,3):
        raise ValueError("n_components must be either 2 or 3.")
    embeddings = UMAP(n_components=n_components, **kwargs).fit_transform(embeddings)

    # Get unique labels
    unique_labels, indices = np.unique(labels, return_inverse=True)
    num_labels = len(unique_labels)
    if -1 in unique_labels:     # Do not include noise in the number of labels
        num_labels -= 1
        
    # Initialize plot
    if ax is None:  
        fig = plt.figure()
        proj = '3d' if n_components == 3 else 'rectilinear'
        ax = fig.add_subplot(111, projection=proj)

    # Set up colormap
    cmap = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    norm = colors.Normalize(vmin=0, vmax=num_labels-1)
    scalar_map = mplcm.ScalarMappable(norm=norm, cmap=cmap)
    color_cycle = [scalar_map.to_rgba(i) for i in range(num_labels)]
    if -1 in unique_labels:
        index = np.argwhere(unique_labels==-1)[0][0]
        color_cycle.insert(index, 'gray')       # Add grey for noise
    ax.set_prop_cycle(color=color_cycle)

    # Plot
    for i, label in enumerate(unique_labels):
        ax.scatter(*embeddings[indices==i].T, alpha=alpha)
        ax.set_aspect('equal', 'box')

    return ax