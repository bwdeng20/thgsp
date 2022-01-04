# Installation

## 1. Install PyTorch

We recommend users to install thgsp in an [conda](https://conda.io/docs/user-guide/install/index.html/)
virtual environment. Install PyTorch1.5 or later following [the official instructions](https://pytorch.org/). Type the
following in the terminal to check if PyTorch is installed successfully.

```
python -c "import torch; print(torch.__version__);print(torch.version.cuda)"
>>> 1.10.1  # pytorch version is 1.10.1
>>> 11.3    # my cuda version is 11.3
```

## 2. Install PyTorch Extensions

Matthias Fey provides excellent PyTorch extensions for graph-related computations. You can install
[pytorch_scatter](https://github.com/rusty1s/pytorch_scatter),
[pytorch_sparse](https://github.com/rusty1s/pytorch_sparse) and
[pytorch_cluster](https://github.com/rusty1s/pytorch_cluster) following the
[official guide](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html). Given `PyTorch1.10.1`
built with `cuda11.3`, the following commands suffice on Linux.

``` shell
# Linux Bash
export CUDA=cu113
export TORCH=1.10.1
pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
``` 

For Windows `CMD`, try:

``` shell
# Windows CMD
set CUDA=cu113
set TORCH=1.10.1
pip install torch-scatter -f https://data.pyg.org/whl/torch-%TORCH%+%CUDA%.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-%TORCH%+%CUDA%.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-%TORCH%+%CUDA%.html
```
---
**NOTE**
`${CUDA}` (or `%CUDA%` on Windows) and `${TORCH}` (or `%TORCH%` on Windows) should be replaced by a specific CUDA 
version (`cpu`, `cu111`, `cu113`) and PyTorch version (`1.8.1`, `1.9.0`, `1.9.1`, `1.10.0`, `1.10.1`), respectively. 
For example, for PyTorch 1.10.1 and CUDA 11.3, type: 
```
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.1+cu113.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.10.1+cu113.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-1.10.1+cu113.html
```
---
## 3. Install thgsp

### Installation via Prebuilt Pip Wheels
We provide prebuilt pip wheels for your convenience. Please check them on [my website](https://wheel.torchgsp.xyz/whl).
You can install thgsp wheels on Linux via the block below.

``` shell
# Linux Bash
export CUDA=cu113
export TORCH=1.10.1
pip install thgsp -f https://wheel.torchgsp.xyz/whl/torch-${TORCH}+${CUDA}
```
For Windows `CMD`, use:
```
# Windows CMD
set CUDA=cu113
set TORCH=1.10.1
pip install thgsp  -f https://wheel.torchgsp.xyz/whl/torch-%TORCH%+%CUDA%
```

### Installation from Source

You can install it from source. Clone the thgsp repository from  `github`.

```
git clone git@github.com:bwdeng20/thgsp.git # from github
```

Build thgsp from source, and this may take many minutes to build `C++` extensions.

```
cd thgsp
pip install .
```

## 4. Install cupy for linear algebra on GPU (Optional)

If you do NOT have an Nvidia GPU, please skip this section.

Follow the [official instruction](https://docs.cupy.dev/en/stable/install.html) to install
`CuPy`, a GPU-based `NumPy` and `SciPy`, via either `pip` or `conda`. Given `CUDA v11.3`, 
the following command works.
```
pip install cupy-cuda113
```