import numpy as np
from scipy.spatial.distance import cdist
import networkx as nx
from matplotlib import pyplot as plt

k = 5
n = 50

X = np.random.rand(n, 48)
distances = cdist(X,X)
core_dist = np.partition(distances, k-1, axis=1)[:,k-1]
mreach_dist = np.clip(np.maximum.outer(core_dist, core_dist), distances, None)


def draw_MST(MST, filename):
    pos = nx.spring_layout(MST)
    nx.draw_networkx(MST, pos)
    labels = nx.get_edge_attributes(MST, 'weight')
    nx.draw_networkx_edge_labels(MST, pos, edge_labels=labels)
    plt.savefig(filename)
    plt.close()


def insert_to_MST(z):
    new_MST = nx.Graph()
    new_MST.add_nodes_from(MST.nodes)
    edge_weights = mreach_dist
    visited = np.zeros(len(MST.nodes), dtype=bool)

    t = None
    def insert(r):
        nonlocal t
        visited[r] = True
        new_edge = (r, z)
        for w in MST.neighbors(r):
            if not visited[w]:
                insert(w)
                smallest_edge, largest_edge = sorted([(r,w), t], key=lambda x: edge_weights[x])
                new_MST.add_edge(*smallest_edge, weight=edge_weights[smallest_edge])
                # draw_MST(new_MST, 'new_MST.png')
                if edge_weights[largest_edge] < edge_weights[new_edge]:
                    new_edge = largest_edge
        t = new_edge

    insert(0)
    new_MST.add_edge(*t, weight=edge_weights[t])    # Need to add the last edge

    return new_MST


def reconstruct_MST(affected):
    for i in affected:




## Test against the true MST
MST = nx.minimum_spanning_tree(nx.Graph(mreach_dist[:n-1][:,:n-1]))
new_MST = insert_to_MST(n-1)
# draw_MST(MST, 'MST.png')
# draw_MST(new_MST, 'new_MST.png')
true_MST = nx.minimum_spanning_tree(nx.Graph(mreach_dist))
try:
    assert nx.is_isomorphic(true_MST, new_MST, edge_match=lambda x,y: x['weight'] == y['weight'])
    print('Test passed')
except AssertionError:
    true_nodes = sorted(true_MST.nodes)
    my_nodes = sorted(new_MST.nodes)
    try:
        assert np.allclose(true_nodes, my_nodes)
        print('Node Test passed')
    except:
        print('Test failed: nodes are not the same')
        print('True nodes:', true_nodes)
        print('My nodes:', my_nodes)

    true_edges = set(true_MST.edges)
    my_edges = set(new_MST.edges)
    try:
        assert true_edges == my_edges
        print('Edge Test passed')
    except:
        print('Edge Test failed: edges are not the same, checking the difference...')
        for e in true_edges.symmetric_difference(my_edges):
            print(f"{e}: \t{mreach_dist[e]}")
        print('If pairs of edges have the same weight, the MSTs are equivalent.')

    
## Test decrement of edge weight
MST = nx.minimum_spanning_tree(nx.Graph(mreach_dist))
np.pad(mreach_dist, ((0,1),(0,1)), 'constant', constant_values=np.inf)




