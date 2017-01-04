"""
This file contains several algorithms which are based on the Matrix Product State (MPS) ansatz.

* Infinite Size Density Matrix Renormalization Group (iDMRG)
* Infinte Time Evolution Bond Decimation (iTEBD)
* Finite Size Density Matrix Renormalization Group (fDMRG)
* Finte Time Evolution Bond Decimation (fTEBD)

"""

import numpy as np

class iDMRG:
  def __init__(self,Gs,SVMs,s,chi,tolerence):
    """
    Define the global objects of iDMRG class.
    """
    self.Gs=Gs
    self.SVMs=SVMs
    self.d=d
    self.chi=chi
    self.tolerence=tolerence
    
  def effH(self):
    
  def update_MPS(self):  

    
class iTEBD:
  def __init__(self):
    
  def time_evolution(self):   
    
    
    
class fDMRG:
  def __init__(self):
  
  def effH(self):
  
  def variational_update(self):
  
  
  
class fTEBD:
  def __init__(self):
