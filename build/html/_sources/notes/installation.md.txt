# Installation

## 1. Install PyTorch

We recommend users to install thgsp in an [conda](https://conda.io/docs/user-guide/install/index.html/)
virtual environment. Install PyTorch1.5 or later following [the official instructions](https://pytorch.org/). Type the
following in the terminal to check if PyTorch is installed successfully.

```
python -c "import torch; print(torch.__version__);print(torch.version.cuda)"
>>> 2.5.1  # pytorch version is 2.5.1
>>> 12.1  # my cuda version is 12.1
```

## 2. Install PyTorch Extensions

Matthias Fey provides excellent PyTorch extensions for graph-related computations. You can install
[pytorch_scatter](https://github.com/rusty1s/pytorch_scatter),
[pytorch_sparse](https://github.com/rusty1s/pytorch_sparse) and
[pytorch_cluster](https://github.com/rusty1s/pytorch_cluster) following the
[official guide](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html). 
Briefly speaking, given `PyTorch2.5.1` built with `cuda12.1`, the following commands finish the 
PyTorch extension installation on Linux.

``` shell
# Linux Bash
export CUDA=cu121
export TORCH=2.5.1
pip install torch-scatter torch-sparse   torch-cluster -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
``` 

For Windows `CMD`, try:

``` shell
# Windows CMD
set CUDA=cu121
set TORCH=2.5.1
pip install torch-scatter  torch-sparse  torch-cluster  -f https://data.pyg.org/whl/torch-%TORCH%+%CUDA%.html
```
---
**NOTE**
`${CUDA}` (or `%CUDA%` on Windows) and `${TORCH}` (or `%TORCH%` on Windows) should be replaced by a specific CUDA 
version (`cpu`, `cu121`) and PyTorch version (`1.8.1`, `2.5.1`), respectively. 
For example, for PyTorch 2.5.1 and CUDA 12.1, type: 
```
pip install torch-scatter torch-sparse   torch-cluster  -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
```
---
## 3. Install thgsp ⭐ 

### Installation via Prebuilt Pip Wheels
We provide prebuilt pip wheels for your convenience. Please check them on [my website](https://wheel.torchgsp.xyz/whl).
You can install thgsp wheels on Linux via the block below.

``` shell
# Linux Bash
export CUDA=cu121
export TORCH=2.5.1
pip install thgsp -f https://wheel.torchgsp.xyz/whl/torch-${TORCH}+${CUDA}
```
For Windows `CMD`, use:
```
# Windows CMD
set CUDA=cu121
set TORCH=2.5.1
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
`CuPy`, cuda-based  `NumPy` and `SciPy`, via either `pip` or `conda`. Given `CUDA 12.1`, 
the following command works.
```
pip install cupy-cuda12x
```