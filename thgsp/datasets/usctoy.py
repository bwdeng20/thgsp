import os
from thgsp.graphs import Graph
from thgsp.io import loadmat
from torchvision.datasets.utils import download_and_extract_archive, check_integrity
from .utils import get_data_dir_of_thgsp


class Toy:
    """
    USC toy graph by STAC group(https://github.com/STAC-USC/GraphStructures).

    Parameters
    ----------
    root:   str, optional
        The root directory to place the downloaded files. If :obj:`None`, set the root dir as "thgsp.datasets.data".
    download: bool
        If True, download the raw .zip file and process it.

    """
    zip_md5 = "ecd24f2683add0f7d37a15e63abdc79e"
    filename = "GraphStructures-master.zip"
    url = 'https://codeload.github.com/STAC-USC/GraphStructures/zip/refs/heads/master'
    toy_mat_md5 = "25ac08c89bb268bb26b9bcff8d21ed18"
    toy_mat_dir = os.path.join("GraphStructures-master", "ToyGraphGSPExample")
    toy_mat_fname = "toy_graph.mat"

    def __init__(self, root=None, download=False):
        self.root = get_data_dir_of_thgsp() if root is None else root
        self.path = os.path.join(self.root, self.toy_mat_dir, self.toy_mat_fname)
        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        data = loadmat(self.path)["toy_graph"][0, 0]
        A = data[0]
        self.A = A
        self.xy = data[2]  # np.dtype('<f8') == np.dtype('float64')
        self.distances = data[3]

    def __getitem__(self, idx):
        assert idx == 0
        return Graph(self.A, coords=self.xy, distances=self.distances)

    def _check_integrity(self) -> bool:
        return check_integrity(self.path, self.toy_mat_md5)

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, self.root, self.filename, self.zip_md5, remove_finished=True)
