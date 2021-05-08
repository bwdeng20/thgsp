# Installation

## 1.Requirements

### 1.1 Install PyTorch

We recommend users to install pytorch_gsp in an [conda](https://conda.io/docs/user-guide/install/index.html/)
virtual environment. Install PyTorch1.5 or later
following [the official instructions](https://pytorch.org/). Type the following in the terminal to check if PyTorch
is installed successfully.
```
python -c "import torch; print(torch.__version__)"
>>> 1.7.0  # here my pytorch version is 1.7.0 
```

### 1.2 Install PyTorch Extensions

Matthias Fey provides excellent PyTorch extensions for graph-related computations. Please install 
[pytorch_scatter](https://github.com/rusty1s/pytorch_scatter) and 
[pytorch_cluster](https://github.com/rusty1s/pytorch_cluster) following the 
[official guide](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).
Remember **NOT** to install [pytorch_sparse](https://github.com/rusty1s/pytorch_sparse) 
before installing [METIS](http://glaros.dtc.umn.edu/gkhome/metis/metis/download).
#### 1.2.1 Install torch_sparse

There is one bipartite graph approximation algorithm in **thgsp** requiring 
[METIS](http://glaros.dtc.umn.edu/gkhome/metis/metis/download) to partition graphs. You need to following the 
instructions in `Install.txt` file to install it. Then, set an environment variable to inform `torch_sparse` to build 
with `METIS` support.

```
export WITH_METIS=1 # for linux
pip install torch-sparse
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

## 2.Install thgsp
### 2.1 From source

You can only install it from source at present. 

1 . Clone the thgsp repository from  `github`.

```
git clone git@github.com:bwdeng20/thgsp.git # from github
```

2 . build thgsp from source.

```
cd thgsp
python setup.py
```