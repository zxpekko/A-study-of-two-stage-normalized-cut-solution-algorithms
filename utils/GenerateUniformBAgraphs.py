from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
import networkx as nx
import pickle
class EdgeType(Enum):

    UNIFORM = 1
    DISCRETE = 2
    RANDOM = 3
class GraphGenerator(ABC):

    def __init__(self, n_spins, edge_type, biased=False):
        self.n_spins = n_spins
        self.edge_type = edge_type
        self.biased = biased

    def pad_matrix(self, matrix):
        dim = matrix.shape[0]
        m = np.zeros((dim+1,dim+1))
        m[:-1,:-1] = matrix
        return matrix

    def pad_bias(self, bias):
        return np.concatenate((bias,[0]))

    @abstractmethod
    def get(self, with_padding=False):
        raise NotImplementedError
class RandomBarabasiAlbertGraphGenerator(GraphGenerator):

    def __init__(self, n_spins=20, m_insertion_edges=4, edge_type=EdgeType.DISCRETE):
        super().__init__(n_spins, edge_type, False)

        self.m_insertion_edges = m_insertion_edges

        if self.edge_type == EdgeType.UNIFORM:
            self.get_connection_mask = lambda : np.ones((self.n_spins,self.n_spins))
        elif self.edge_type == EdgeType.DISCRETE:
            def get_connection_mask():
                mask = 2. * np.random.randint(2, size=(self.n_spins, self.n_spins)) - 1.
                mask = np.tril(mask) + np.triu(mask.T, 1)
                return mask
            self.get_connection_mask = get_connection_mask
        elif self.edge_type == EdgeType.RANDOM:
            def get_connection_mask():
                mask = 2.*np.random.rand(self.n_spins,self.n_spins)-1
                mask = np.tril(mask) + np.triu(mask.T, 1)
                return mask
            self.get_connection_mask = get_connection_mask
        else:
            raise NotImplementedError()

    def get(self, with_padding=False,random_seed=1):

        g = nx.barabasi_albert_graph(self.n_spins, self.m_insertion_edges,seed=random_seed)
        adj = np.multiply(nx.to_numpy_array(g), self.get_connection_mask())

        # No self-connections (this modifies adj in-place).
        np.fill_diagonal(adj, 0)

        return self.pad_matrix(adj) if with_padding else adj