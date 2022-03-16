import primme
import numpy as np
import scipy.linalg as spla
from scipy.sparse.linalg import LinearOperator
from typing import Tuple, Union


def svd(matrix: np.ndarray, cutoff: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """

    Args:
        matrix: Input Matrix.
        cutoff: Truncation dimensions.

    Returns:

    """
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    u = u[:, 0:cutoff]
    s = s[0:cutoff]
    vt = vt[0:cutoff, :]
    return u, s, vt


def qr(matrix: np.ndarray, cutoff: int) -> Tuple[np.ndarray, np.ndarray]:
    """

    Args:
        matrix: Input Matrix.
        cutoff: Truncation dimensions.

    Returns:

    """
    q, r = np.linalg.qr(matrix, mode='reduced')
    q = q[:, 0:cutoff]
    r = r[0:cutoff, :]
    return q, r


def eigh(matrix: np.ndarray, k: int = 1,
         backend: str = 'numpy', **kwargs) -> Tuple[Union[float, np.ndarray], np.ndarray]:
    """

    Args:
        matrix: Hermitian or real symmetric matrices whose eigenvalues and eigenvectors are to be computed.
        k: The number of eigenpairs to be computed.
        backend: ('numpy' or 'scipy').
        **kwargs: Keyword arguments for scipy solver.

    Returns:

    """
    evals, evecs = {
        'numpy': np.linalg.eigh(matrix),
        'scipy': spla.eigh(matrix, **kwargs)
    }[backend]
    return (evals[0], evecs[:, 0]) if k == 1 else (evals[:k], evecs[:, :k])


def eigshmv(linear_operator: LinearOperator, v0: np.ndarray,
            k: int = 1, which: str = 'SA', tol: float = 0,
            **kwargs) -> Tuple[Union[float, np.ndarray], np.ndarray]:
    """

    Args:
        linear_operator:
        v0: Initial guesses to the eigenvectors.
        k: The number of eigenpairs to be computed.
        which: Which k eigenvectors and eigenvalues to find.
        tol: Tolerance for eigenpairs (stopping criterion). The default value is sqrt of machine precision.
        **kwargs: Keyword arguments for primme solver.

    Returns:

    """
    evals, evecs = primme.eigsh(linear_operator, v0=v0, k=k, which=which, tol=tol, **kwargs)
    return (evals[0], evecs) if k == 1 else (evals, evecs)
