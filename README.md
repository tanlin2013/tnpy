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
1. Declare your Tensor Network State and initialize it if you want
   
   ```
   import TNpy
   myMPS=TNpy.tnstate.MPS(whichMPS,d,chi,N)
   Gs=mymps.initialize()
   ```
2. Customize your own Matrix Product Operator for the desired model.

   ```
   Class XXZ:
       def __int__(self,N,delta):
           self.N=N
           self.delta=delta
           
       def M(self,site):
           MPO=TNpy.operators.MPO(whichMPS='f',N=self.N,D=5)
           Sp,Sm,Sz,I2,O2=TNpy.operators.spin_operators()
           
           elem=[[I2,-0.5*Sp,-0.5*Sm,-self.delta*Sz,O2],
              [O2,O2,O2,O2,Sm],
              [O2,O2,O2,O2,Sp],
              [O2,O2,O2,O2,Sz],
              [O2,O2,O2,O2,I2]]
           
           M=MPO.assign_to_MPO(elem,site)
           return M
   ```
3. Call the algorithm to optimize the state. 
   
   ```
   simulation=TNpy.algorithm.fDMRG(model.M,Gs,N,d,chi)
   E,stats=simulation.variational_optimize()
   Gs=simulation.Gs
   ```
4. Compute any physical quantities whatever you want from the obtained state. Those common measurements may be avaliable inside `/src/TNpy/measurement.py`.
