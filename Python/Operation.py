"""
This file contains the fundamental functions for the Matrix Product State (MPS) operations. 
"""

import numpy as np
from scipy import sparse
import Primme

class MPS:   
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
        
        Gs=[] ; if svm: SVMs=[]   
        if whichMPS=='i':
            for i in range(2):
                Gs.append(np.random.rand(self.chi,self.d,self.chi))
                if svm: SVMs.append(np.diagflat(np.random.rand(self.chi)))
        elif whichMPS=='f':
            size_parity=size%2
            for site in range(size):        
                if size_parity==0:
                    if site==0 or site==size-1:
                        self.Gs.append(np.random.rand(self.d,min(self.d,self.chi)))
                    elif site<=size/2-1 and site!=0:
                        self.Gs.append(np.random.rand(min(self.d**site,self.chi),self.d,min(self.d**(site+1),self.chi)))
                    elif site>size/2-1 and site!=size-1:
                        self.Gs.append(np.random.rand(min(self.d**(size-site),self.chi),self.d,min(self.d**(size-1-site),self.chi)))
                elif size_parity==1:
                    if site==0 or site==size-1:
                        self.Gs.append(np.random.rand(self.d,min(self.d,self.chi)))
                    elif site<size/2 and site!=0:
                        self.Gs.append(np.random.rand(min(self.d**site,self.chi),self.d,min(self.d**(site+1),self.chi)))
                    elif site==size/2:
                        self.Gs.append(np.random.rand(min(self.d**site,self.chi),self.d,min(self.d**site,self.chi)))
                    elif site>size/2 and site!=size-1:
                        self.Gs.append(np.random.rand(min(self.d**(size-site),self.chi),self.d,min(self.d**(size-1-site),self.chi)))
        else:
            raise ValueError('only iMPS and fMPS are supported.')        
        if svm:
            return Gs,SVMs
        else:
            return Gs 
        
    def initialize_EnvLs(self,whichMPS):

        return
    
    def initialize_EnvRs(self,whichMPS):
    
        return    

class contraction: 
    def __int__(self):
        
    def transfer_operator(self):    
        return
    
    def update_EnvLs(self):
        return

    def update_EnvRs(self):
        return
        
def eigensolver(H,psi):
    """
    This function is a warpper of PRIMME function eigsh().
    
    * Parameters:
        * H: ndarray
        * psi: array
    * Returns:
        * evals:
        * evecs:
    """
    A=sparse.csr_matrix(H)
    evals,evecs,stats=Primme.eigsh(A,k=1,which='SA',v0=psi,tol=self.tolerance,return_stats=True)                                    
    return evals[0],evecs

def Trotter_Suzuki_Decomposition():
    """
    """
    return

def inverse_SVM(A):
    """
    Compute the inverse of Singular Value Matrix.
    
    * Parameters:
        A: ndarray
            The Singular Value Matrix wants to be convert.
    * Returns:
        A_inv: ndarray
            The inverse of Singular Value Matrix.
    """
    
    return A_inv
