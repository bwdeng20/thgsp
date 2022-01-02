# Installation

## 1. Pip

### Install PyTorch

We recommend users to install thgsp in an [conda](https://conda.io/docs/user-guide/install/index.html/)
virtual environment. Install PyTorch1.5 or later following [the official instructions](https://pytorch.org/). Type the
following in the terminal to check if PyTorch is installed successfully.

```
python -c "import torch; print(torch.__version__)"
>>> 1.10.0  # here my pytorch version is 1.8.0 
```

### Install PyTorch Extensions

Matthias Fey provides excellent PyTorch extensions for graph-related computations. Please install
[pytorch_scatter](https://github.com/rusty1s/pytorch_scatter),
[pytorch_sparse](https://github.com/rusty1s/pytorch_sparse) and
[pytorch_cluster](https://github.com/rusty1s/pytorch_cluster) following the
[official guide](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html). Given `PyTorch1.10.0`
with `cudatoolkit11.1`, the following commands suffice.

``` shell
# Linux Bash
export CUDA=cu111
export TORCH=1.10.0
pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
``` 
For Windows `CMD`, try commands below
```cmd
set CUDA=cu111
set TORCH=1.10.0
pip install torch-scatter -f https://data.pyg.org/whl/torch-%TORCH%+%CUDA%.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-%TORCH%+%CUDA%.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-%TORCH%+%CUDA%.html
```

### Install cupy for linear algebra on GPU (Optional)

If you do NOT have an Nvidia GPU, please skip this section.

Follow the [official instruction](https://docs.cupy.dev/en/stable/install.html) to install
`CuPy`, a GPU-based `NumPy` and `SciPy`, via either `pip` or `conda`. Given `cudatoolkit11.1`, 
the following command is suitable.

```
pip install cupy-cuda111
```

### From source

You can install it from source

Clone the thgsp repository from  `github`.

```
git clone git@github.com:bwdeng20/thgsp.git # from github
```

Build thgsp from source, and this may take many minutes.

```
cd thgsp
pip install .
```


### Prebuilt Wheels
We provide prebuilt pip wheels for your convenience. Please check them on this [website](http://16.162.201.90/whl/).
You have to check your `cuda`, `torch`, `python` versions and the `OS` to find a wheel matching your case.