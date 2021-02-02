import numpy as np
import torch
from scipy.sparse import coo_matrix

from thgsp.graphs import Graph
from thgsp.io import loadmat


class Minnesota:
    """
    Minnesota traffic network

    Parameters
    ----------
    connected: bool
        If True, connect the traffic network
    """

    def __init__(self, connected=True):
        data = loadmat("minnesota")
        self.is_connected = connected
        A = data['A']
        if connected:
            A[348, 354] = 1
            A[354, 348] = 1
        self.A = coo_matrix(A.astype(np.float64))
        self.xy = data['xy'].astype(np.float64)
        self.f = torch.from_numpy(data['f']).reshape(-1)
        self.F = np.squeeze(data['F']).astype(int)

    def __getitem__(self, idx):
        assert idx == 0
        return Graph(self.A, coords=self.xy)
