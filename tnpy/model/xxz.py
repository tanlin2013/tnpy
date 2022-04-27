import numpy as np

from tnpy.operators import SpinOperators
from tnpy.model.model_1d import Model1D
from tnpy.model.utils import boundary_vectors


class XXZ(Model1D):
    def __init__(self, n: int, delta: float):
        """

        Args:
            n: System size.
            delta: Coupling strength on z direction.
        """
        super().__init__(n)
        self.delta = delta

    @boundary_vectors(row=0, col=-1)
    def _elem(self, site: int) -> np.ndarray:
        Sp, Sm, Sz, I2, O2 = SpinOperators()
        return np.array(
            [
                [I2, -0.5 * Sp, -0.5 * Sm, -self.delta * Sz, O2],
                [O2, O2, O2, O2, Sm],
                [O2, O2, O2, O2, Sp],
                [O2, O2, O2, O2, Sz],
                [O2, O2, O2, O2, I2],
            ]
        )
