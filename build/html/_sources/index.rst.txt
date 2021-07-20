:github_url: https://github.com/bwdeng20/thgsp

Welcome to thgsp's documentation!
**********************************

Thgsp is a graph signal processing package based on `PyTorch <https://pytorch.org/>`_.

The repository now mainly includes:

-  GFT-based filter(banks) processing multi-dimensional signals in a Multiple Input Multiple Output(MIMO) manner.
-  GraphQmf and GraphBiorth wavelet filter bank.
-  Many strategies to decompose an arbitrary graph into many(usually <10) bipartite graphs.
-  Many graph signal sampling(which differs slightly with
   `general graph sampling <https://github.com/benedekrozemberczki/littleballoffur>`_ ) and reconstruction algorithms.

As this package is built on `PyTorch <https://pytorch.org/>`_  and
`pytorch_sparse <https://github.com/rusty1s/pytorch_sparse>`_, you can easily integrate functionalities here into a
PyTorch pipeline.

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Notes

   notes/installation
   notes/introduction
   notes/waveletfb


.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Package Reference

   modules/graphs
   modules/bga
   modules/filters
   modules/alg
   modules/datasets
   modules/sampling

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
