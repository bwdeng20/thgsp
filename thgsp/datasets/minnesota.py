import os
import torch
import numpy as np
from os.path import join
from shutil import move
from scipy.sparse import coo_matrix
from thgsp.graphs import Graph
from thgsp.io import loadmat
from torchvision.datasets.utils import download_and_extract_archive, check_integrity
from .utils import get_data_dir_of_thgsp, remove_file_or_dir


class Minnesota:
    r"""
    The Minnesota traffic network shipped with "Wavelet Filterbanks on Graph" [1]_ project(with a license of GPL V3)
    by USC-STAC group(see http://biron.usc.edu/wiki/index.php/Graph_Filterbanks). This road network is a part of
    "The National Highway Planning Network (NHPN) 2000-Present dataset" [3]_ and also carried by MatlabBGL [2]_
    toolbox(released under BSD).

    Parameters
    ----------
    root:   str, optional
        The root directory to place the downloaded files. If :obj:`None`, set the root dir as "thgsp.datasets.data".
    connected: bool
        If True, connect the traffic network.
    download: bool
        If True, download the raw .zip file and process it.

    References
    ----------
    .. [1]  S K. Naran, et al, "Perfect Reconstruction Two-channel Wavelet Filter Banks for Graph Structured Data",
            IEEE trans on Signal Processing, 2012.
    .. [2] D. Gleich, "https://github.com/dgleich/matlab-bgl", GitHub, 2011
    .. [3] https://www.fhwa.dot.gov/planning/processes/tools/nhpn/

    """
    filename = "Graph_Wavelets_Demo.zip"
    zip_md5 = "83bf2aef1ad7e75badfc6296ca336d44"
    url = "http://sipi.usc.edu/~ortega/Software/Graph_Wavelets_Demo.zip"
    files2keep = ["LICENSE.txt"]
    top_dir = "minnesota-usc"
    mat_list = [
        ["min_traffic_graph.mat", "d9b45cd25dca3997c8454da0efa01926"],
        ["min_coloring.mat", "a6c75092048f5ab0409c6c951435924e"],
        ["min_graph_signal.mat", "14efc15373708b63cd7fbaec8ad7284f"],
    ]

    def __init__(self, root=None, connected=True, download=False):
        self.root = get_data_dir_of_thgsp() if root is None else root

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted."
                + " You can use download=True to download it"
            )
        g = loadmat(join(self.root, self.top_dir, self.mat_list[0][0]))
        A = g["A"]
        self.is_connected = connected
        if connected:
            A[348, 354] = 1
            A[354, 348] = 1
        self.A = coo_matrix(A.astype(np.float64))
        self.xy = g["xy"].astype(np.float64)
        F = loadmat(join(self.root, self.top_dir, self.mat_list[1][0]))["F"]
        self.F = np.squeeze(F).astype(int) - 1

        f = loadmat(join(self.root, self.top_dir, self.mat_list[2][0]))["f"]
        self.f = torch.from_numpy(f).reshape(-1)

    def __getitem__(self, idx):
        assert idx == 0
        return Graph(self.A, coords=self.xy)

    def _check_integrity(self):
        root = self.root
        for mat_fname, md5sub in self.mat_list:
            fpath = join(root, self.top_dir, mat_fname)
            if not check_integrity(fpath, md5sub):
                return False
        return True

    def download(self):
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        root = self.root
        download_and_extract_archive(
            self.url, root, root, self.filename, self.zip_md5, remove_finished=True
        )
        os.rename(join(root, "Graph Wavelets Demo"), join(root, "minnesota-usc"))

        files2keep = self.files2keep
        files2keep.append("Datasets")
        dirs = os.listdir(join(root, self.top_dir))
        for file_or_dir in dirs:
            if file_or_dir not in files2keep:
                remove_file_or_dir(join(root, self.top_dir, file_or_dir))

        for mat_file, md5sub in self.mat_list:
            abs_path = join(root, self.top_dir, "Datasets", mat_file)
            check_integrity(abs_path, md5sub)
            move(abs_path, join(root, self.top_dir, mat_file))

        remove_file_or_dir(join(root, self.top_dir, "Datasets"))
