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
