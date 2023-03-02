from functools import wraps
from typing import Callable

import numpy as np


def boundary_vectors(row: int, col: int) -> Callable:
    """
    The boundary vectors on two ends are usually just certain row and column in
    the Matrix Product Operator (MPO). This decorator is used to specified
    which row and column are the boundary vectors.

    Args:
        row: By which row in MPO is the boundary vector on the left end.
        col: By which column in MPO is the boundary vector on the right end.

    Returns:

    """

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(self, site: int) -> np.ndarray:
            tensor = func(self, site)
            if site == 0:
                tensor = tensor[row, :, :, :]
            elif site == self.n - 1:
                tensor = tensor[:, col, :, :]
            return tensor

        return wrapper

    return decorator


def minors_if_no_penalty(row: int, col: int) -> Callable:
    """
    If the model definition comes with penalty term,
    but the user set the penalty strength to zero,
    that means there are a lots of unnecessary zeros presenting in
    the Matrix Product Operator (MPO).
    This decorator can help to eliminate the zeros row and column,
    without touching the definition of penalty MPO.

    Args:
        row: By which row in MPO is the penalty term.
        col: By which column in MPO is the penalty term.

    Returns:

    """

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(self, *args, **kwargs) -> np.ndarray:
            mat = func(self, *args, **kwargs)
            if self.penalty == 0:
                return np.delete(np.delete(mat, row, axis=0), col, axis=1)
            else:
                return mat

        return wrapper

    return decorator
