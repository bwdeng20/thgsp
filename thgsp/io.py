import io
import pkgutil

import scipy.io


def loadmat(path):
    r"""
    Load a matlab data file.

    Parameters
    ----------
    path : string
        Path to the mat file from the data folder, without the .mat extension.

    Returns
    -------
    data : dict
        dictionary with variable names as keys, and loaded matrices as
        values.
    """
    try:
        data = pkgutil.get_data(
            'thgsp', 'datasets/data/pointclouds/' + path + '.mat')
    except FileNotFoundError:
        data = open(path, 'rb').read()
    data = io.BytesIO(data)
    return scipy.io.loadmat(data)
