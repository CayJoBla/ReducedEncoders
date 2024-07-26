import numpy as np

class UnionFind:
    def __init__(self, initial_weights):
        N = len(initial_weights)
        self.parent = np.arange(2*N-1, dtype=int)
        self.weight = np.hstack((np.array(initial_weights), np.zeros(N-1)))
        self.next_label = N

    def union(self, i, j):
        self.parent[i] = self.next_label
        self.parent[j] = self.next_label
        self.weight[self.next_label] = self.weight[i] + self.weight[j]
        self.next_label += 1
        return self.next_label - 1  # Return the label of the union

    def find(self, i):
        j = i
        while self.parent[i] != i:
            i = self.parent[i]
        while self.parent[j] != i:
            self.parent[j], j = i, self.parent[j]
        return i


class TreeUnionFind:
    def __init__(self, size):
        self.size = np.zeros(size, dtype=int)
        self.parent = np.arange(size, dtype=int)
        self.is_component = np.ones(size, dtype=bool)

    def union(self, i, j):
        i_root = self.find(i)
        j_root = self.find(j)

        if self.size[i_root] < self.size[j_root]:
            self.parent[i_root] = j_root
        if self.size[i_root] > self.size[j_root]:
            self.parent[j_root] = i_root
        else:
            self.parent[j_root] = i_root
            self.size[i_root] += 1

        return

    def find(self, i):
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
            self.is_component[i] = False
        return self.parent[i]

    def components(self):
        return self.is_component.nonzero()[0]