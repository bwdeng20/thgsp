from thgsp.graphs import Graph
from thgsp.io import loadmat


class Toy:
    """
    USC toy graph

    """

    def __init__(self):
        data = loadmat("toy_graph")
        A = data['A']
        self.A = A
        self.xy = data['layout']  # np.dtype('<f8') == np.dtype('float64')
        self.distances = data['distances']

    def __getitem__(self, idx):
        assert idx == 0
        return Graph(self.A, coords=self.xy, distances=self.distances)
