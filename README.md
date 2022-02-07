# tnpy

![build](https://github.com/tanlin2013/tnpy/actions/workflows/build.yml/badge.svg)
![tests](https://github.com/tanlin2013/tnpy/actions/workflows/tests.yml/badge.svg)
![docs](https://github.com/tanlin2013/tnpy/actions/workflows/docs.yml/badge.svg)
![license](https://img.shields.io/github/license/tanlin2013/tnpy?style=plastic)

[Documentation](https://tanlin2013.github.io/tnpy/) |

This project is a python implementation of Tensor Network,
a numerical approach to quantum many-body systems.
  

**tnpy** is built on top of [quimb](https://github.com/jcmgray/quimb) for tensor contractions, 
with optimized support for various backend engines (TensorFlow, JAX, PyTorch, and Numpy). 
For eigen-solver we adopt [primme](https://github.com/primme/primme),
an iterative multi-method solver with preconditioning.

Currently, we support Matrix Product State (MPS) algorithms, 
with more are coming...

* Finite-sized Density Matrix Renormalization Group (fDMRG)
* Tree tensor Strong Disorder Renormalization Group (tSDRG)

fDMRG & tSDRG are on alpha-release. 
For others, please expect edge cases.

Requirments
-----------
See `requirements.txt` for more details.
But these two are essential building blocks.

  * [quimb](https://github.com/jcmgray/quimb)
  * [primme/primme](https://github.com/primme/primme)

Regarding any installation problems with Primme,
please refer to [Primme official](http://www.cs.wm.edu/~andreas/software/). 
Also, it's required to have [lapack](http://www.netlib.org/lapack/) and [blas](http://www.netlib.org/blas/)
installed in prior to Primme.

Installation
------------

   * using Docker
     
     ```
     docker run --rm -it tanlin2013/tnpy
     ```
   * using pip:
     
     ```
     pip install git+https://github.com/tanlin2013/tnpy@main
     ```
   
Documentation
-------------
For details about **tnpy**, see the [reference documentation](https://tanlin2013.github.io/tnpy/).
    
Getting started
---------------
1. Defining the Matrix Product Operator of your model as a Callable function with argument `site` of the type `int`, 
   e.g. the function `_elem(self, site)` below. 
   The MPO class then accepts the Callable as an input and constructs a MPO object. 

   ```
   import numpy as np
   from tnpy.operators import SpinOperators, MPO
   from tnpy.finite_dmrg import FiniteDMRG
   
   class XXZ:

       def __init__(self, N: int, delta: float) -> None:
           self.N = N
           self.delta = delta

       def _elem(self, site: int) -> np.ndarray:
           Sp, Sm, Sz, I2, O2 = SpinOperators()
           return np.array(
               [[I2, -0.5 * Sp, -0.5 * Sm, -self.delta * Sz, O2],
                [O2, O2, O2, O2, Sm],
                [O2, O2, O2, O2, Sp],
                [O2, O2, O2, O2, Sz],
                [O2, O2, O2, O2, I2]]
           )
        
       @property
       def mpo(self) -> MPO:
           return MPO(self.N, self._elem)
   ```
   
2. Call the algorithm to optimize the state. 
   
   ```
   N = 100  # length of spin chain
   chi = 60  # virtual bond dimension 
   
   model = XXZ(N, delta)
   fdmrg = FiniteDMRG(
       mpo=model.mpo,
       chi=chi
   )
   fdmrg.update(tol=1e-8)
   ```
   
3. Compute any physical quantities whatever you want from the obtained state.
   The resulting MPS is of the type `tensornetwork.FiniteMPS`,
   see [here](https://tensornetwork.readthedocs.io/en/latest/stubs/tensornetwork.FiniteMPS.html#tensornetwork.FiniteMPS) for more details.
   
   ```
   my_mps = fdmrg.mps
   ```
