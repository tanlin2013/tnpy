.. tnpy documentation master file, created by
   sphinx-quickstart on Wed Mar 17 13:01:44 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to tnpy's documentation!
================================

.. mdinclude:: ../../README.md

Algorithms
----------
For the moment, we concern the one-dimensional algorithms only.
These include,

.. toctree::
   :numbered:
   :maxdepth: 2

   algorithm/exact_diagonalization
   algorithm/finite_dmrg
   algorithm/finite_tdvp
   algorithm/tsdrg

Built-in models
---------------
We provide the following built-in models in 1D.

.. toctree::
   :numbered:
   :maxdepth: 2
   :caption:

   model/model_1d
   model/xxz
   model/dimer_xxz
   model/random_heisenberg
   model/thirring

Operators and linear algebra
----------------------------

.. toctree::
   :numbered:
   :maxdepth: 2
   :caption:

   matrix_product_state
   operators
   linalg

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
