# Installation

## 1. Pip

### 1.1 Install PyTorch

We recommend users to install pytorch_gsp in an [conda](https://conda.io/docs/user-guide/install/index.html/)
virtual environment. Install PyTorch1.5 or later following [the official instructions](https://pytorch.org/). Type the
following in the terminal to check if PyTorch is installed successfully.

```
python -c "import torch; print(torch.__version__)"
>>> 1.8.0  # here my pytorch version is 1.8.0 
```

### 1.2 Install PyTorch Extensions

Matthias Fey provides excellent PyTorch extensions for graph-related computations. Please install
[pytorch_scatter](https://github.com/rusty1s/pytorch_scatter),
[pytorch_sparse](https://github.com/rusty1s/pytorch_sparse) and
[pytorch_cluster](https://github.com/rusty1s/pytorch_cluster) following the
[official guide](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html). Given `PyTorch1.8.x`
with `cudatoolkit11.1`, the following commands suffice.

```
export CUDA=cu111
export TORCH=1.8.0
pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
```

### 1.3 Install SuiteSparse for scikit-sparse

On Debian/Ubuntu systems, the following command should suffice:

```
sudo apt-get install libsuitesparse-dev
```

On Arch (Manjaro) Linux, run:

```
sudo pacman -S suitesparse
```

### 1.4 Install cupy for linear algebra on GPU (Optional)

If you do NOT have an Nvidia GPU, please skip this section.

Follow the [official instruction](https://docs.cupy.dev/en/stable/install.html) to install
`CuPy`, a GPU-based `NumPy` and `SciPy`, via either `pip` or `conda`. Given `PyTorch1.8.x` with `cudatoolkit11.1`, 
the following command is suitable.

```
conda install -c conda-forge cupy cudatoolkit=11.1
```

### 1.5 From source

You can only install it from source at present since it's pre-released.

Clone the thgsp repository from  `github`.

```
git clone git@github.com:bwdeng20/thgsp.git # from github
```

Build thgsp from source, and this may take many minutes.

```
cd thgsp
pip install .
```