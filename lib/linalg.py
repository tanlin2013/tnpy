import numpy as np
import Primme


def svd(A, chi):
    u, s, vt = np.linalg.svd(A, full_matrices=False)
    u = u[:, 0:chi]
    s = s[0:chi]
    vt = vt[0:chi, :]
    return u, s, vt


def eigshmv(Afunc, v0, k=1, sigma=None, which='SA',
            ncv=None, maxiter=None, tol=0, return_eigenvectors=True,
            Minv=None, OPinv=None, mode='normal', lock=None,
            return_stats=False, maxBlockSize=0, minRestartSize=0,
            maxPrevRetain=0, method=2):
    class PP(Primme.PrimmeParams):
        def __init__(self):
            Primme.PrimmeParams.__init__(self)

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
    evecs = np.zeros((pp.n, pp.numOrthoConst + pp.numEvals), dtype, order='F')

    if lock is not None:
        np.copyto(evecs[:, 0:pp.numOrthoConst], lock[:, 0:pp.numOrthoConst])

    if v0 is not None:
        pp.initSize = min(v0.shape[1], pp.numEvals)
        np.copyto(evecs[:, pp.numOrthoConst:pp.numOrthoConst + pp.initSize],
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
