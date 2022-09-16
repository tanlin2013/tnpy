import numpy as np

from tnpy.operators import SpinOperators
from tnpy.model.model_1d import Model1D
from tnpy.model.utils import boundary_vectors


class TransverseIsing(Model1D):
    def __init__(self, n, j, h):
        r"""

        .. math::

            H = -j \left( \sum_{i=0}^{N-1} S_{i+1}^z S_i^z + h S_i^x \right)

        Args:
            n: System size.
            j: Overall prefactor of energy.
            h: The coupling strength of transversed field.
        """
        super().__init__(n)
        self._j = j
        self._h = h

    @property
    def j(self):
        return self._j

    @property
    def h(self):
        return self._h

    @boundary_vectors(row=0, col=-1)
    def _elem(self, site: int) -> np.ndarray:
        Sp, Sm, Sz, I2, O2 = SpinOperators()
        Sx = Sp + Sm
        return np.array(
            [[I2, -self.j * Sz, -self.j * self.h * Sx], [O2, O2, Sz], [O2, O2, I2]],
            dtype=float,
        )
