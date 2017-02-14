import numpy as np
from scipy import sparse
import Primme

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
