import matplotlib.pyplot as plt
import numpy as np
import torch

from thgsp.datasets.utils import get_data_dir_of_thgsp, os, remove_file_or_dir
from thgsp.utils import mse, snr


def get_number_env(key_str, default_val=None):
    key_val = os.environ.get(key_str)
    if key_val is None:
        key_val = default_val
        return key_val
    # key_val is str
    if key_val.isdigit():
        nbr = int(key_val)
    else:
        nbr = float(key_val)
    return nbr


def to_tensor(x, dtype, device=None):
    return None if x is None else torch.as_tensor(x, dtype=dtype, device=device)


def plot(plotly_fig=None):
    show_flag = os.environ.get("THGSP_PLT")
    show_flag = 0 if show_flag is None else int(show_flag)
    if show_flag:
        if plotly_fig is not None:
            plotly_fig.show()
        plt.plot()


def remove_downloaded_dataset(name):
    dire = get_data_dir_of_thgsp()
    remove_file_or_dir(os.path.join(dire, name))


def snr_and_mse(x, target):
    s, m = snr(x, target), mse(x, target)
    print(f"SNR: {s:.4f} | MSE: {m:.4f}")
    return s, m


np.set_printoptions(linewidth=10000, precision=4)
torch.set_printoptions(linewidth=10000, precision=4)

dtypes = [torch.float, torch.double, torch.int, torch.long]

grad_dtypes = [torch.float, torch.double]
float_dtypes = grad_dtypes
float_np_dts = [np.float32, np.float64]
int_dtypes = [torch.int, torch.long]
sparse_formats = ("csc", "csr", "coo")
lap_types = ["comb", "sym", "rw", None]

color_strategies = ["harary", "osglm"]
num_strategies = ["admm", "amfs"]

devices = [torch.device("cpu")]
if torch.cuda.is_available():
    devices += [torch.device(f"cuda:{torch.cuda.current_device()}")]

partition_strategy = ["graclus", "metis"]

RAY_NUM_CPUS = get_number_env("RAY_NUM_CPUS", 2)
RAY_NUM_GPUS = get_number_env("RAY_NUM_GPUS")
