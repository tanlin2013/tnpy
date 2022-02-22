import numpy as np
from functools import wraps
from typing import Callable


def boundary_vectors(row: int, col: int) -> Callable:
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
