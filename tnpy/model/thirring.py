import numpy as np
from tnpy.model import ModelBase
from tnpy.operators import SpinOperators
from .utils import boundary_vectors, minors_if_no_penalty


class Thirring(ModelBase):

    def __init__(self, n: int, delta: float, ma: float, penalty: float, s_target: int) -> None:
        """

        Args:
            n: System size.
            delta: Wavefunction-renormalized bare coupling, Delta(g).
            ma: Wavefunction-renormalized bare mass.
            penalty: Penalty strength (of Lagrangian multiplier).
            s_target: The targeting total Sz charge sector.
        """
        super(Thirring, self).__init__(n)
        self.delta = delta
        self.ma = ma
        self.penalty = penalty
        self.s_target = s_target

    @boundary_vectors(row=0, col=-1)
    @minors_if_no_penalty(row=3, col=3)
    def _elem(self, site: int) -> np.ndarray:
        Sp, Sm, Sz, I2, O2 = SpinOperators()

        beta = self.delta + ((-1.0) ** site * self.ma) - 2.0 * self.penalty * self.s_target
        gamma = self.penalty * (0.25 + self.s_target ** 2 / self.n) + 0.25 * self.delta

        return np.array(
            [[I2, -0.5 * Sp, -0.5 * Sm, 2.0 * np.sqrt(self.penalty) * Sz, self.delta * Sz, gamma * I2 + beta * Sz],
             [O2, O2, O2, O2, O2, Sm],
             [O2, O2, O2, O2, O2, Sp],
             [O2, O2, O2, I2, O2, np.sqrt(self.penalty) * Sz],
             [O2, O2, O2, O2, O2, Sz],
             [O2, O2, O2, O2, O2, I2]],
            dtype=float
        )
