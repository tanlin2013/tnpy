"""
This file contains the fundamental functions for the Matrix Product State (MPS) operations. 
"""

import numpy as np
from scipy import sparse
import Primme

class MPS:   
    def __init__(self,whichMPS,d,chi):
        """
        * Parameters:
            * whichMPS: string, {'i','f'} 
                If whichMPS='i', an infinite MPS is initialized. Otherwise if whichMPS='f', a finite MPS is created.  
            * d: int
                The physical bond dimension which is associated to the dimension of single-particle Hilbert space.
            * chi: int
                The visual bond dimension to be keep after the Singular Value Decomposition (SVD).               
        """
        if whichMPS!='i' and whichMPS!='f':
            raise ValueError('Only iMPS and fMPS are supported.')
        self.whichMPS=whichMPS
        self.d=d
        self.chi=chi
        
    def initialize_MPS(self,canonical_form='R',N=None):
        """
        Randomly initialize the MPS.
    
        * Parameters:          
            * canonical_form: string, {'L','R','GL'}, optional
                If whichMPS='f', fMPS can be represented as left-normalized, right-normalized or the standard (Gamma-Lambda representation) MPS.
            * N: int, optional
                If whichMPS='f', the size of system N is needed. 
        
        * Returns: 
            * Gs: list of ndarray
                A list of rank-3 tensors which represents the MPS. The order of tensor is (chi,d,chi) or (d,chi) for the boundaries if fMPS is considered.  
            * SVMs: list of ndarray
                A list of singular value matrices. SVMs is always return for iMPS. But, for the fMPS SVMs only return when canonical_form='GL'.
        """
        
        """ Check the input variables"""
        if self.whichMPS=='f': 
            if not cononical_form in ['L','R','GL'] or type(N) is not int:
                raise ValueError('canonical_form and size must be specified when whichMPS='f'.')        
        
        Gs=[] ; SVMs=[]
        if self.whichMPS=='i':
            """ Create the iMPS """
            for site in xrange(2):
                Gs.append(np.random.rand(self.chi,self.d,self.chi))
                SVMs.append(np.diagflat(np.random.rand(self.chi)))
            return Gs,SVMs    
        elif self.whichMPS=='f':
            """ Create the fMPS in the standard (GL) representation """
            for site in xrange(N):
                if site==0 or site==N-1:
                    Gs.append(np.random.rand(self.d,self.chi))
                else:
                    Gs.append(np.random.rand(self.chi,self.d,self.chi))
                if site<N-1:
                    SVMs.append(np.diagflat(np.random.rand(self.chi)))
            """ Left- or right-normalized the MPS """
            if canonical_form=='L':
                Gs=self.normalize_MPS(Gs,SVMs,order='L')
                return Gs
            elif canonical_form=='R':
                Gs.self.normalize_MPS(Gs,SVMs,order='R')
                return Gs
            elif canonical_form=='GL':                
                return Gs,SVMs
            else:
                raise ValueError('Only the standard (GL), Left- and Right-normalized canonical form are supported.')          
    
    def normalize_MPS(self,Gs,SVMs,order):
        """
        Left or Right normalize the fMPS which is in the standard (GL) representation.
        
        * Parameters:
            * Gs: list of ndarray
                The fMPS wants to be left- or right-normalized.  
            * SVMs: list of ndarray
                The fMPS wants to be left- or right-normalized.
            * order: string, {'L','R'}
                Specified the direction of normalization.
        * Returns:
            * Gs: list of ndarray
                Left- or right-normalized MPS.
        """
        N=len(Gs)
        if order=='R':
            Gs=Gs.reverse()
            SVMs=SVMs.reverse()
        elif order!='L':
            raise ValueError('The order must be either L or R.')
        for site in xrange(N-1):
            if site==0:
                theta=np.tensordot(Gs[site],SVMs[site],axes=(1,0))
            else:
                theta=np.tensordot(Gs[site],SVMs[site],axes=(2,0))
                theta=np.ndarray.reshape(theta,(self.d*Gs[site].shape[0],Gs[site].shape[2]))
        X,S,Y=np.linalg.svd(theta,full_matrices=False)
        if site==N-2:
            Gs[site+1]=np.tensordot(Gs[site+1],np.dot(np.diagflat(S),Y),axes=(1,1))
        else:
            Gs[site+1]=np.tensordot(np.dot(np.diagflat(S),Y),Gs[site+1],axes=(1,0))
        if site==0:
            Gs[site]=np.ndarray.reshape(X,(self.d,Gs[site].shape[1]))
        else:
            Gs[site]=np.ndarray.reshape(X,(Gs[site].shape[0],self.d,Gs[site].shape[2]))
        if order=='R':
            return Gs.reverse()
        else:
            return Gs
    
    def to_GL_rep(self,Gs,direction):
        
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
    evals,evecs=Primme.eigsh(A,k=1,which='SA',v0=psi,tol=1e-12)                                    
    return evals[0],evecs

def svd(A,chi):
    """
    This function is a wrapper of PRIMME svd().
    """
    u,s,vt=Primme.svd(A,k=chi)
    return u,s,vt

def Trotter_Suzuki_Decomposition(h,order):
    """
    """
    
    return

def inverse_SVM(A):
    """
    Compute the inverse of singular value matrix.
    
    * Parameters:
        A: ndarray
            The singular value matrix wants to be convert.
    * Returns:
        A_inv: ndarray
            The inverse of singular value matrix.
    """
    A=np.diag(A)
    A_inv=np.zeros(len(A))
    for i in xrange(len(A)):        
        if A[i]==0:
            A_inv[i]=0.0
        else:
            A_inv[i]=1.0/A[i]
        A_inv=np.diagflat(A_inv) 
    return A_inv

def transfer_operator(Gs,A):
    if Gs.ndim==2:
        trans=np.tensordot(Gs,np.tensordot(A,np.conjugate(Gs),axes=(2,0)),axes=(0,0))
    else:                        
        trans=np.tensordot(Gs,np.tensordot(A,np.conjugate(Gs),axes=(2,1)),axes=(1,0))
        trans=np.swapaxes(trans,1,2)
        trans=np.swapaxes(trans,3,4)
        trans=np.swapaxes(trans,2,3)
    return trans    
