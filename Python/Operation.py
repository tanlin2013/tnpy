"""
This file contains the fundamental functions for the Matrix Product State (MPS) operations. 
"""

import numpy as np
from scipy import sparse
import Primme

class operation:
    
    def __init__(self,d,chi):
        """
        * Parameters:
            * d: int
                The physical bond dimension which is associated to the dimension of single-particle Hilbert space.
            * chi: int
                The visual bond dimension to be keep after the Singular Value Decomposition (SVD).               
        """
        self.d=d
        self.chi=chi
        
    def initialize_MPS(self,whichMPS,svm,canonical_form,size):
        """
        Randomly initialize the MPS.
    
        * Parameters:
            * whichMPS: string, {'i','f'} 
                If whichMPS='i', an infinite MPS is initialized. Otherwise if whichMPS='f', a finite MPS is created.            
            * svm: bool
                If Ture, the Singular Value Matrices will be returned. Else the Singular Value Matrices will be multiplied into MPS.
            * canonical_form: string, {'L','R'}, optional
                If whichMPS='f', fMPS can be either Left-normalized or Right-normalized.
            * size: int, optional
                If whichMPS='f', the system size is needed. 
        
        * Returns: 
            * Gs: list of ndarray
                A list of rank-3 tensors which represents the MPS. The order of tensor is (chi,d,chi) or (d,chi) if the boundaries are considered.  
            * SVMs: list of ndarray
                A list of Singular Value Matrices.
        """
        
        Gs=[]    
        if whichMPS=='i':
            for i in range(2):
                Gs.append(np.random.rand(self.chi,self.d,self.chi))
            if svm:
                SVMs=[]
                for i in range(2):
                    SVMs.append(np.diagflat(np.random.rand(self.chi)))
        elif whichMPS=='f':
            
        else:
            raise ValueError('only iMPS and fMPS are supported.')
        
        if svm:
            return Gs,SVMs
        else:
            return Gs 
        
    def initialize_EnvLs(self):

        return
    
    def initialize_EnvRs(self):
    
        return
    
    def 

