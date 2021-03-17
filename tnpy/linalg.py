import numpy as np
import primme
from scipy.sparse.linalg import LinearOperator
from typing import Callable


def svd(A: np.ndarray, chi: int):
    """

    Args:
        A: Input Matrix
        chi: Truncated dimension

    Returns:

    """
    u, s, vt = np.linalg.svd(A, full_matrices=False)
    u = u[:, 0:chi]
    s = s[0:chi]
    vt = vt[0:chi, :]
    return u, s, vt


def qr(A: np.ndarray, chi: int):
    q, r = np.linalg.qr(A, mode='reduced')
    q = q[:, 0:chi]
    r = r[0:chi, :]
    return q, r


def eigshmv(Afunc: Callable, v0: np.ndarray, tol=0):
    """

    Args:
        Afunc:
        v0:
        tol:

    Returns:

    """
    A = LinearOperator((v0.shape[0], v0.shape[0]), matvec=Afunc)
    evals, evecs = primme.eigsh(A, k=1, which='SA', v0=v0, tol=tol)
    return evals[0], evecs


# def eigshmv(Afunc: Callable,
#             v0: np.ndarray,
#             k=1,
#             sigma=None,
#             which='SA',
#             ncv=None,
#             maxiter=None,
#             tol=0,
#             return_eigenvectors=True,
#             Minv=None,
#             OPinv=None,
#             mode='normal',
#             lock=None,
#             return_stats=False,
#             maxBlockSize=0,
#             minRestartSize=0,
#             maxPrevRetain=0,
#             method=2):
#
#     class PP(Primme.PrimmeParams):
#         def __init__(self):
#             Primme.PrimmeParams.__init__(self)
#
#         def matvec(self, X):
#             return Afunc(X)
#
#         def prevec(self, X):
#             return OPinv.matmat(X)
#
#     pp = PP()
#
#     pp.n = v0.shape[0]
#
#     if k <= 0 or k > pp.n:
#         raise ValueError("k=%d must be between 1 and %d, the order of the "
#                          "square input matrix." % (k, pp.n))
#     pp.numEvals = k
#     pp.correctionParams.precondition = 0 if OPinv is None else 1
#
#     if which == 'LM':
#         pp.target = Primme.primme_largest_abs
#         if sigma is None:
#             sigma = 0.0
#     elif which == 'LA':
#         pp.target = Primme.primme_largest
#         sigma = None
#     elif which == 'SA':
#         pp.target = Primme.primme_smallest
#         sigma = None
#     elif which == 'SM':
#         pp.target = Primme.primme_closest_abs
#         if sigma is None:
#             sigma = 0.0
#     else:
#         raise ValueError("which='%s' not supported" % which)
#
#     if sigma is not None:
#         pp.targetShifts = np.array([sigma], dtype=np.dtype('d'))
#
#     pp.eps = tol
#
#     if ncv is not None:
#         pp.maxBasisSize = ncv
#
#     if maxiter is not None:
#         pp.maxMatvecs = maxiter
#
#     if OPinv is not None:
#         OPinv = sparse.linalg.interface.aslinearoperator(OPinv)
#         if OPinv.shape[0] != OPinv.shape[1] or OPinv.shape[0] != v0.shape[0]:
#             raise ValueError('OPinv: expected square matrix with same shape as A (shape=%s)' % (OPinv.shape,))
#         pp.correctionParams.precondition = 1
#
#     if lock is not None:
#         if lock.shape[0] != pp.n:
#             raise ValueError('lock: expected matrix with the same columns as A (shape=%s)' % (lock.shape,))
#         pp.numOrthoConst = min(lock.shape[1], pp.n)
#
#     dtype = np.dtype("d")
#
#     evals = np.zeros(pp.numEvals)
#     norms = np.zeros(pp.numEvals)
#     evecs = np.zeros((pp.n, pp.numOrthoConst + pp.numEvals), dtype, order='F')
#
#     if lock is not None:
#         np.copyto(evecs[:, 0:pp.numOrthoConst], lock[:, 0:pp.numOrthoConst])
#
#     if v0 is not None:
#         pp.initSize = min(v0.shape[1], pp.numEvals)
#         np.copyto(evecs[:, pp.numOrthoConst:pp.numOrthoConst + pp.initSize],
#                   v0[:, 0:pp.initSize])
#
#     if maxBlockSize:
#         pp.maxBlockSize = maxBlockSize
#
#     if minRestartSize:
#         pp.minRestartSize = minRestartSize
#
#     if maxPrevRetain:
#         pp.restartingParams.maxPrevRetain = maxPrevRetain
#
#     if method is not None:
#         pp.set_method(method)
#
#     err = Primme.dprimme(evals, evecs, norms, pp)
#
#     if err != 0:
#         raise Primme.PrimmeError(err)
#
#     evecs = evecs[:, pp.numOrthoConst:]
#     if return_stats:
#         stats = dict((f, getattr(pp.stats, f)) for f in [
#             "numOuterIterations", "numRestarts", "numMatvecs",
#             "numPreconds", "elapsedTime", "estimateMinEVal",
#             "estimateMaxEVal", "estimateLargestSVal"])
#         return evals[0], evecs, stats
#     else:
#         return evals[0], evecs


class KrylovExpm:

    def __init__(self, delta, mat, v0, n=20):
        self.delta = delta
        self.mat = mat
        self.v0 = v0 / np.linalg.norm(v0)
        self.n = n
        assert(n <= mat.shape[0])

    def orthonormalize(self, vecs):
        pass

    def construct_krylov_space(self):
        vecs = [self.v0]
        for i in range(self.n-1):
            vecs.append(np.dot(self.mat, vecs[-1]))
        # vecs = self.orthonormalize(vecs)
        print(len(vecs))
