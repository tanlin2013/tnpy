import primme
import numpy as np
from scipy.sparse.linalg import LinearOperator
from typing import Callable


def svd(matrix: np.ndarray, chi: int):
    """

    Args:
        matrix: Input Matrix
        chi: Truncated dimension

    Returns:

    """
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    u = u[:, 0:chi]
    s = s[0:chi]
    vt = vt[0:chi, :]
    return u, s, vt


def qr(matrix: np.ndarray, chi: int):
    q, r = np.linalg.qr(matrix, mode='reduced')
    q = q[:, 0:chi]
    r = r[0:chi, :]
    return q, r


def eigshmv(matvec_func: Callable, v0: np.ndarray, tol=0):
    """

    Args:
        matvec_func:
        v0:
        tol:

    Returns:

    """
    lin_op = LinearOperator((v0.shape[0], v0.shape[0]), matvec=matvec_func)
    evals, evecs = primme.eigsh(lin_op, k=1, which='SA', v0=v0, tol=tol)
    return evals[0], evecs
