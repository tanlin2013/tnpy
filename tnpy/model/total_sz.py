import numpy as np
from tnpy.model import ModelBase
from tnpy.operators import SpinOperators
from .utils import boundary_vectors


class TotalSz(ModelBase):

    def __init__(self, n: int):
        super(TotalSz, self).__init__(n)

    @boundary_vectors(row=0, col=-1)
    def _elem(self, site: int) -> np.ndarray:
        Sp, Sm, Sz, I2, O2 = SpinOperators()
        return np.array(
            [[I2, Sz],
             [O2, I2]],
            dtype=float
        )
