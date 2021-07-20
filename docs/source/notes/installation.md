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
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
```

### 1.3 Install SuiteSparse for scikit-sparse

On Debian/Ubuntu systems, the following command should suffice:

```
sudo apt-get install libsuitesparse-dev
```

On Arch Linux, run:

```
sudo pacman -S suitesparse
```

### 1.4 Install cupy for linear algebra on GPU(Optional)

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

## 2. Docker

We also provide docker images for thgsp. Check [here](https://hub.docker.com/r/bowen20/thgsp)
for all available tags.

```
docker pull bowen20/thgsp:v0.11-dev
```

All images are integrated with ssh services to ease the usage. In general, you should expose container ports
(especially `22`)and map them to those of the host. If some external data(e.g., datasets) are in need, one has to mount
host volume to the container. An example command is below

```
docker run -it --gpus all -v  /home/USER_NAME:/workspace -v /data/datasets：/datasets --name thgspdev -p 2222:22 -p 7777:8888 --restart always thgsp:v0.11-dev
```

With this line, the host paths `/home/USER_NAME` and `/data/datasets` are mapped into `/workspace` and `/datasets` of
container named `thgspdev`, separately. In addition, container ports `22` and `8888` are bind to host ports
`2222` and `7777`, respectively. The port `22` is for `ssh` service, and the password for the default user `root` is
`106996`.

Once container started, one can use the internal conda python interpreter located at container
path `/opt/conda/bin/python`.