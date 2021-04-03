import numpy as np
from tnpy.operators import SpinOperators, MPO


class XXZ:

    def __init__(self, N: int, delta: float) -> None:
        self.N = N
        self.delta = delta

    def _elem(self, site: int) -> np.ndarray:
        Sp, Sm, Sz, I2, O2 = SpinOperators()
        return np.array(
            [[I2, -0.5 * Sp, -0.5 * Sm, -self.delta * Sz, O2],
             [O2, O2, O2, O2, Sm],
             [O2, O2, O2, O2, Sp],
             [O2, O2, O2, O2, Sz],
             [O2, O2, O2, O2, I2]]
        )

    def mpo(self) -> MPO:
        return MPO(self.N, self._elem)
