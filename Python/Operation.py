"""
This file contains the fundamental functions for Tensor Network operations. 
"""

import numpy as np
from scipy import sparse
import Primme

class operation:
    
    def __init__(self,d,chi):
        """
        # Parameters:
            d: float
                The physical bond dimension which is associated to the dimension of single-particle Hilbert space.
            chi: float
                The visual bond dimension to be keep after the Singular Value Decomposition (SVD).               
        """
        self.d=d
        self.chi=chi
        
    def initialize_MPS(self,whichMPS,svm,size):
        """
        Randomly initialize the MPS.
    
        # Parameters:
            whichMPS: string, {'i','f'} 
                If whichMPS='i', an infinite MPS is initialized. Otherwise if whichMPS='f', a finite MPS is created.  
            svm: bool
                Specify whether the Singular Value Matrices are needed. If Ture, the Singular Value Matrices will be returned. 
            size: int, optional
                If size='f', the system size is needed. 
        
        # Returns: 
            Gs: list of ndarray
                
            Ls: list of ndarray
                
        """
    
        return

def initialize_EnvLs():

    return
    
def initialize_EnvRs():
    
    return

