# TNpy

This project is a python implementation of Tensor Network,
a numerical approaches to quantum many-body system.
  

TNpy is built on top of [google/TensorNetwork](https://github.com/google/TensorNetwork) for tensor contractions, 
with optimized support for various backend engines (TensorFlow, JAX, PyTorch, and Numpy). 
For eigen-solver we adopt [primme](https://github.com/primme/primme),
an iterative multi-method solver with preconditioning.

Currently, we supports Matrix Product State (MPS) algorithms, 
with more are coming...
 
* Infinite Size Density Matrix Renormalization Group (iDMRG)
* Infinte Time Evolution Bond Decimation (iTEBD)
* Finite Size Density Matrix Renormalization Group (fDMRG)
* Time Dependent Variational Principle (TDVP) (ongoing...)

fDMRG is on alpha-release and is much stable. 
For others, please expect edge cases.

## Requirments:

See `requirements.txt` for more details.
But these two are essential building blocks.

  * [google/TensorNetwork](https://github.com/google/TensorNetwork)
  * [primme/primme](https://github.com/primme/primme)

Regarding any installation problems with Primme,
please refer to [Primme official](http://www.cs.wm.edu/~andreas/software/). 

## Installation
Simply run the file `setup.py` with command:    
```
python setup.py install    
```
Or, if you are using Docker container
```
make build & make run
```
   
## Documentation
For details about TNpy, see the [reference documentation](https://tanlin2013.github.io/TNpy/).
    
## How to use it?
1. Defining your model's Hamiltonian in Matrix Product Operator (MPO).
 What you have to do is to write a function which depends on `site` and retuns a np.ndarray that represents your MPO.

   ```
   import numpy as np
   from TNpy.operators import SpinOperators, MPO
   from TNpy.finite_dmrg import FiniteDMRG
   
   Class XXZ:
       def __int__(self, N, delta):
           self.N = N
           self.delta = delta
           
       def mpo(self, site):
           Sp, Sm, Sz, I2, O2 = SpinOperators()
           
           return np.array(
                [[I2,-0.5*Sp,-0.5*Sm,-self.delta*Sz,O2],
                [O2,O2,O2,O2,Sm],
                [O2,O2,O2,O2,Sp],
                [O2,O2,O2,O2,Sz],
                [O2,O2,O2,O2,I2]])
   ```
2. Call the algorithm to optimize the state. 
   
   ```
   N = 100  # length of spin chain
   chi = 60  # virtual bond dimension 
   
   model = XXZ(N, delta)
   fdmrg = FiniteDMRG(mpo=MPO(N, model.mpo), chi=chi)
   fdmrg.update(tol=1e-8)
   ```
3. Compute any physical quantities whatever you want from the obtained state.
