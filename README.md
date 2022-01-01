# *thgsp*

![](./docs/source/_static/img/logo.png) \
[![Github Build Status](https://github.com/bwdeng20/thgsp/actions/workflows/building.yml/badge.svg)](http://16.162.201.90/whl/)
[![Github Doc Status](https://github.com/bwdeng20/thgsp/actions/workflows/doc.yml/badge.svg)](https://bwdeng20.github.io/thgsp)
[![codecov](https://codecov.io/gh/bwdeng20/thgsp/branch/main/graph/badge.svg?token=H45AEGPM0P)](https://codecov.io/gh/bwdeng20/thgsp)
[![Linting](https://github.com/bwdeng20/thgsp/actions/workflows/linting.yml/badge.svg)]()
[![Linting](https://github.com/bwdeng20/thgsp/actions/workflows/testing.yml/badge.svg)]()
[![License](https://img.shields.io/static/v1?label=license&message=BSD&color=red)](https://github.com/bwdeng20/thgsp/blob/main/LICENSE) 
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) \
A **g**raph **s**ignal **p**rocessing toolbox built on [py**t**orc**h**](https://github.com/pytorch/pytorch). The repository
now mainly consists of the following stuffs:

1. GFT-based filter(banks) processing multi-dimensional signals in a Multiple Input Multiple Output(MIMO) manner.
2. GraphQmf and GraphBiorth wavelet filter bank.
3. Many strategies to decompose an arbitrary graph into many(usually <10) bipartite graphs.
4. Many graph signal sampling(which differs slightly with
   [general graph sampling](https://github.com/benedekrozemberczki/littleballoffur>)) and reconstruction algorithms.

As this package is built on [PyTorch](https://pytorch.org)  and
[pytorch_sparse](https://github.com/rusty1s/pytorch_sparse>), you can easily integrate functionalities from
**thgsp** into a PyTorch pipeline. Check the [document](https://bwdeng20.github.io/thgsp/)
for **installation** and introduction.

## Table of Contents

- [Example](#example)
- [Reference](#reference)
- [Citation](#citation)

## Example

### GraphQMF four channel wavelet filter bank on Minnesota

<img src="./demos/minnesota/TheWaveletCoefficientsFourChannels.png" width="50%" height="40%">
<img src="./demos/minnesota/ReconstructedFourChannels.png" width="50%" height="40%">


The Minnesota traffic network is 3-colorable(exactly) or 4-colorable(roughly). Hence 4-channel GraphQmf filterbank is
constructed, requiring a ceil(log2(4))=2 level bipartite decomposition. The bipartite graphs are below.

<img src="./demos/minnesota/TowBipartiteGraphs.png" width="50%" height="40%">

The comparision between the eventual reconstructed signal and the input one.

<img src="./demos/minnesota/Signals.png " width="50%" height="40%">

### GraphBiorth four channel wavelet filter bank for camera man.

See the full program [here](./demos/images/bio_cmn.py).

## Reference

[David K Hammond, et al.] [Wavelets on Graphs via Spectral Graph Theory](https://arxiv.org/abs/0912.3848) \
[Sunil K. Narang, et al.] [Compact Support Biorthogonal Wavelet Filterbanks for Arbitrary Undirected Graphs](https://ieeexplore.ieee.org/document/6557512) \
[Sunil K. Narang, et al.] [Perfect Reconstruction Two-Channel Wavelet Filter Banks for Graph Structured Data](https://ieeexplore.ieee.org/document/6156471) \
[Akie Sakiyama, et al.] [Oversampled Graph Laplacian Matrix for Graph Filter Banks](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6937182) \
[Jing Zen, et al.] [Bipartite Subgraph Decomposition for Critically Sampledwavelet Filterbanks on Arbitrary Graphs](https://ieeexplore.ieee.org/document/7472871) \
[Aamir Anis, et al.] [Towards a Sampling Theorem for Signals on Arbitrary Graphs](https://ieeexplore.ieee.org/document/6854325) \
[Aimin Jiang, et al.] [ Admm-based Bipartite Graph Approximation](https://ieeexplore.ieee.org/document/8682548/) \
[Yuanchao Bai, et al.] Fast graph sampling set selection using Gershgorin disc alignment, IEEE TSP, 2020 \
[G. Puy, et al.] Random sampling of bandlimited signals on graphs,  ACHA, 2018. \
[A. Sakiyama, et al.] Eigendecomposition-free sampling set selection for graph signals，IEEE TSP, 2020. \
[Aamir Anis et al.] Efficient sampling set selection for bandlimited graph signals using graph spectral proxies, IEEE TSP, 2016.

## Citation
```
@misc{thgsp,
  author = {Bowen Deng},
  title = {ThGSP: A PyTorch-based Graph Signal Processing Library},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/bwdeng20/thgsp}},
}
```
