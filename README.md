# TNpy
This project contains several algorithms which are based on the Matrix Product State (MPS) ansatz, and is used for the studies of (1+1) dimensional physics. 

* Infinite Size Density Matrix Renormalization Group (iDMRG)
* Infinte Time Evolution Bond Decimation (iTEBD)
* Finite Size Density Matrix Renormalization Group (fDMRG)
* Finte Time Evolution Bond Decimation (fTEBD)

This work is still in progress...

## Requirments:
  * Numpy
  * Scipy  
  * PRIMME
  
Regarding any installation problems with PRIMME, please refer http://www.cs.wm.edu/~andreas/software/. 

## Installation
  Simply run the file `setup.py` with the command:
  ```
  python setup.py install
  ```
         
## How to use it?
1. Declare your Tensor Network State via
   
   ```
   import TNpy
   myMPS=TNpy.tnstate.MPS(whichMPS,d,chi,N)
   ```
2. Customize your own Matrix Product Operator for the desired model.


3. Call the algorithm to optimize the state. 

