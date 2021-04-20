import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from thgsp.datasets.utils import remove_file_or_dir, get_data_dir_of_thgsp, os

dtypes = [torch.float, torch.double, torch.int, torch.long]

grad_dtypes = [torch.float, torch.double]
float_dtypes = grad_dtypes
float_np_dts = [np.float32, np.float64]
int_dtypes = [torch.int, torch.long]

lap_types = ['comb', 'sym', 'rw', None]

color_strategies = ["harary", "osglm"]
num_strategies = ["admm", "amfs"]

devices = [torch.device('cpu')]
if torch.cuda.is_available():
    devices += [torch.device(f'cuda:{torch.cuda.current_device()}')]

partition_strategy = ["graclus", 'metis']


def to_tensor(x, dtype, device=None):
    return None if x is None else torch.as_tensor(x, dtype=dtype, device=device)


def plot(plotly_fig=None):
    show_flag = os.environ.get('THGSP_PLT')
    show_flag = 0 if show_flag is None else int(show_flag)
    if show_flag:
        if plotly_fig is not None:
            plotly_fig.show()
        plt.plot()


def remove_downloaded_dataset(name):
    dire = get_data_dir_of_thgsp()
    remove_file_or_dir(os.path.join(dire, name))
