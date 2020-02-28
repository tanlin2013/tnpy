import numpy as np
from scipy import sparse
from scipy.linalg import svd as scipy_svd
from sklearn.utils.extmath import randomized_svd
import Primme

def eigsh(H, psi):
    """
    This function is a warpper of PRIMME function eigsh().
    
    * Parameters:
        * H: ndarray
        * psi: array
    * Returns:
        * evals:
        * evecs:
    """
    A = sparse.csr_matrix(H)
    evals, evecs = Primme.eigsh(A,k=1,which='SA',v0=psi,
                             ncv=None, maxiter=None, tol=0, return_eigenvectors=True,
                             Minv=None, OPinv=None, mode='normal', lock=None,
                             return_stats=False, maxBlockSize=0, minRestartSize=0,
                             maxPrevRetain=0, method=None)                                
    return evals[0],evecs

def eigshmv(Afunc, v0, k=1, sigma=None, which='SA', 
          ncv=None, maxiter=None, tol=0, return_eigenvectors=True,
          Minv=None, OPinv=None, mode='normal', lock=None,
          return_stats=False, maxBlockSize=0, minRestartSize=0,
          maxPrevRetain=0, method=2):

    class PP(Primme.PrimmeParams):
        def __init__(self):
            super(PP, self).__init__()
        def matvec(self, X):
            return Afunc(X)
        def prevec(self, X):
            return OPinv.matmat(X)

    pp = PP()

    pp.n = v0.shape[0]

    if k <= 0 or k > pp.n:
        raise ValueError("k=%d must be between 1 and %d, the order of the "
                         "square input matrix." % (k, pp.n))
    pp.numEvals = k
    pp.correctionParams.precondition = 0 if OPinv is None else 1

    if which == 'LM':
        pp.target = Primme.primme_largest_abs
        if sigma is None:
            sigma = 0.0
    elif which == 'LA':
        pp.target = Primme.primme_largest
        sigma = None
    elif which == 'SA':
        pp.target = Primme.primme_smallest
        sigma = None
    elif which == 'SM':
        pp.target = Primme.primme_closest_abs
        if sigma is None:
            sigma = 0.0
    else:
        raise ValueError("which='%s' not supported" % which)

    if sigma is not None:
        pp.targetShifts = np.array([sigma], dtype=np.dtype('d'))

    pp.eps = tol

    if ncv is not None:
        pp.maxBasisSize = ncv

    if maxiter is not None:
        pp.maxMatvecs = maxiter

    if OPinv is not None:
        OPinv = sparse.linalg.interface.aslinearoperator(OPinv)
        if OPinv.shape[0] != OPinv.shape[1] or OPinv.shape[0] != v0.shape[0]:
            raise ValueError('OPinv: expected square matrix with same shape as A (shape=%s)' % (OPinv.shape,))
        pp.correctionParams.precondition = 1

    if lock is not None:
        if lock.shape[0] != pp.n:
            raise ValueError('lock: expected matrix with the same columns as A (shape=%s)' % (lock.shape,))
        pp.numOrthoConst = min(lock.shape[1], pp.n)

    dtype = np.dtype("d")

    evals = np.zeros(pp.numEvals)
    norms = np.zeros(pp.numEvals)
    evecs = np.zeros((pp.n, pp.numOrthoConst+pp.numEvals), dtype, order='F')

    if lock is not None:
        np.copyto(evecs[:, 0:pp.numOrthoConst], lock[:, 0:pp.numOrthoConst])

    if v0 is not None:
        pp.initSize = min(v0.shape[1], pp.numEvals)
        np.copyto(evecs[:, pp.numOrthoConst:pp.numOrthoConst+pp.initSize],
            v0[:, 0:pp.initSize])

    if maxBlockSize:
        pp.maxBlockSize = maxBlockSize

    if minRestartSize:
        pp.minRestartSize = minRestartSize

    if maxPrevRetain:
        pp.restartingParams.maxPrevRetain = maxPrevRetain

    if method is not None:
        pp.set_method(method)

    err = Primme.dprimme(evals, evecs, norms, pp)
 
    if err != 0:
        raise Primme.PrimmeError(err)

    evecs = evecs[:, pp.numOrthoConst:]
    if return_stats:
        stats = dict((f, getattr(pp.stats, f)) for f in [
            "numOuterIterations", "numRestarts", "numMatvecs",
            "numPreconds", "elapsedTime", "estimateMinEVal",
            "estimateMaxEVal", "estimateLargestSVal"])
        return evals[0], evecs, stats
    else:
        return evals[0], evecs

def svd(A, chi, method='numpy'):
    """
    This function provides several ways to implement svd.
    """
    dim = min(min(A.shape),chi)
    if method == 'primme':
        u, s, vt = Primme.svd(A,k=dim)
    elif method == 'numpy':
        u, s, vt = np.linalg.svd(A,full_matrices=False)
        u = u[:,0:dim]; s = s[0:dim]; vt = vt[0:dim,:]
    elif method == 'scipy':
        u, s, vt = scipy_svd(A,full_matrices=False)
        u = u[:,0:dim]; s = s[0:dim]; vt = vt[0:dim,:]
    elif method == 'scipy_sparse':
        u, s, vt = sparse.linalg.svd(A,k=dim)
    elif method == 'scikit':
        u, s, vt = randomized_svd(A,chi)
    return u, s, vt

"""
def Trotter_Suzuki_Decomposition(h, order):
    
    return
"""

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
    A = np.diag(A)
    A_inv = np.zeros(len(A))
    for i in range(len(A)):        
        if A[i] == 0:
            A_inv[i] = 0.0
        else:
            A_inv[i] = 1.0/A[i]
    A_inv = np.diagflat(A_inv) 
    return A_inv
