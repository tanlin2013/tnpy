# tnpy

_________________

[![PyPI version](https://badge.fury.io/py/tnpy.svg)](http://badge.fury.io/py/tnpy)
[![Docker build](https://github.com/tanlin2013/tnpy/actions/workflows/build.yml/badge.svg)](https://github.com/tanlin2013/tnpy/actions/workflows/build.yml)
[![Test Status](https://github.com/tanlin2013/tnpy/actions/workflows/test.yml/badge.svg)](https://github.com/tanlin2013/tnpy/actions/workflows/test.yml)
[![Lint Status](https://github.com/tanlin2013/tnpy/actions/workflows/lint.yml/badge.svg)](https://github.com/tanlin2013/tnpy/actions/workflows/lint.yml)
[![codecov](https://codecov.io/gh/tanlin2013/tnpy/branch/main/graph/badge.svg)](https://codecov.io/gh/tanlin2013/tnpy)
[![Join the chat at https://gitter.im/tanlin2013/tnpy](https://badges.gitter.im/tanlin2013/tnpy.svg)](https://gitter.im/tanlin2013/tnpy?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![License](https://img.shields.io/github/license/mashape/apistatus.svg)](https://pypi.python.org/pypi/tnpy/)
[![Downloads](https://pepy.tech/badge/tnpy)](https://pepy.tech/project/tnpy)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://timothycrosley.github.io/isort/)
_________________

[Documentation](https://tanlin2013.github.io/tnpy/) |
_________________

This project is a python implementation of Tensor Network,
a numerical approach to quantum many-body systems.

**tnpy** is built on top of [quimb](https://github.com/jcmgray/quimb),
along with [TensorNetwork](https://github.com/google/TensorNetwork)
for tensor contractions, with optimized support for various backend engines
(TensorFlow, JAX, PyTorch, and Numpy).
For eigen-solver we adopt [primme](https://github.com/primme/primme),
an iterative multi-method solver with preconditioning.

Currently, we support Matrix Product State (MPS) algorithms,
with more are coming...

* Exact Diagonalization (ED)
* Finite-sized Density Matrix Renormalization Group (fDMRG)
* Tree tensor Strong Disorder Renormalization Group (tSDRG)

fDMRG & tSDRG are on alpha-release.
For others, please expect edge cases.

Requirements
------------
Dependencies are listed in
[pyproject.toml](https://github.com/tanlin2013/tnpy/blob/main/pyproject.toml),
and they are supposed to be installed together with **tnpy**.
Here we just list the essential building blocks.

* [jcmgray/quimb](https://github.com/jcmgray/quimb)
* [google/Tensornetwork](https://github.com/google/TensorNetwork)
* [primme/primme](https://github.com/primme/primme)

Also, it's required to have [lapack](http://www.netlib.org/lapack/)
and [blas](http://www.netlib.org/blas/) installed in prior to Primme.
They can also be installed through pip
with [mkl-devel](https://pypi.org/project/mkl-devel/).

Installation
------------

* Using Docker

  ```
  docker run --rm -it tanlin2013/tnpy
  ```
* Using pip
    * Latest release:
      ```
      pip install tnpy
      ```
    * Development version:
      ```
      pip install git+https://github.com/tanlin2013/tnpy@main
      ```
* Optional dependencies
    * If [lapack](http://www.netlib.org/lapack/) and
      [blas](http://www.netlib.org/blas/) are missing
      ```
      pip install tnpy[mkl]
      ```
    * For [quimb](https://github.com/jcmgray/quimb) drawing functionality.
      This will install [matplotlib](https://matplotlib.org/)
      and [networkx](https://networkx.org/)
      ```
      pip install tnpy[drawing]
      ```

Getting started
---------------

1. We provide built-in models. Though it's also possible to register your own one.

   ```
   import numpy as np
   from tnpy.finite_dmrg import FiniteDMRG
   from tnpy.model import XXZ

   model = XXZ(n=100, delta=0.5)
   fdmrg = FiniteDMRG(
       mpo=model.mpo,
       chi=60  # virtual bond dimensions
   )
   fdmrg.update(tol=1e-8)
   ```

2. Compute any physical quantities whatever you want from the obtained state.
   The resulting MPS is of the type `quimb.tensor.MatrixProductState`,
   see [here](https://tanlin2013.github.io/tnpy/matrix_product_state.html)
   for more details.

   ```
   my_mps = fdmrg.mps
   ```

License
-------
© Tan Tao-Lin, 2023. Licensed under a [MIT](https://github.com/tanlin2013/tnpy/master/LICENSE)
license.
