"""
This file contains several algorithms which are based on the Matrix Product State (MPS) ansatz.

* Infinite Size Density Matrix Renormalization Group (iDMRG)
* Infinte Time Evolution Bond Decimation (iTEBD)
* Finite Size Density Matrix Renormalization Group (fDMRG)
* Finte Time Evolution Bond Decimation (fTEBD)

"""

import numpy as np
import Operation

class iDMRG:
    def __init__(self,MPO,Gs,SVMs,N,d,D,chi,tolerence):
        """
        Define the global objects of iDMRG class.
        """
        self.MPO=MPO
        self.Gs=Gs
        self.SVMs=SVMs
        self.N=N
        self.d=d
        self.D=D
        self.chi=chi
        self.tolerence=tolerence
    
    def initialize_Env(self):
        vL=np.zeros(self.D)
        vL[0]=1.0
        L=np.kron(vL,np.identity(self.chi,dtype=float))      
        L=np.ndarray.reshape(L,(self.chi,self.D,self.chi))
        vR=np.zeros(D)
        vR[-1]=1.0
        R=np.kron(np.identity(self.chi,dtype=float),vR)
        R=np.ndarray.reshape(R,(self.chi,D,self.chi))
        return L,R 
  
    def warm_up_optimize(self):
        L,R
    
    def convergence(self):  

    
class iTEBD:
    def __init__(self):
    
    def time_evolution(self):   
    
    
    
class fDMRG:
    def __init__(self,MPO,Gs,d,chi,tolerence):
        self.MPO=MPO
  
    def initialize_Env(self,direction):
        if direction=='L':
            L=[]
            for site in range(N-1):
                M=self.MPO(site)
                if site==0:
                    envL=contraction.transfer_operator(M,site)             
                else:    
                    envL=contraction.update_envL(envL,M,site)            
                L.append(envL)   
            return L  
        elif direction='R':    
            return R
    
    def update_Env(self):  
  
    def effH(self):
  
    def variational_optimize(self):
  
    def convergence(self):
  
class fTEBD:
    def __init__(self):
