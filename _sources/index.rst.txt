.. tnpy documentation master file, created by
   sphinx-quickstart on Wed Mar 17 13:01:44 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to tnpy's documentation!
================================

.. mdinclude:: ../../README.md

API references
--------------
For the moment, we concern the one-dimensional algorithms only.
These include,

.. toctree::
   :numbered:
   :maxdepth: 2
   :caption: Algorithms

   algorithm/exact_diagonalization
   algorithm/finite_dmrg
   algorithm/finite_tdvp
   algorithm/tsdrg

.. toctree::
   :numbered:
   :maxdepth: 2
   :caption: Built-in models

   model/model_1d
   model/xxz
   model/dimer_xxz
   model/random_heisenberg
   model/thirring
   model/total_sz
   model/utils

.. toctree::
   :numbered:
   :maxdepth: 2
   :caption: Operators and linear algebra

   matrix_product_state
   operators
   linalg

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
