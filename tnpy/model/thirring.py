import numpy as np
from tnpy.model import ModelBase
from tnpy.operators import SpinOperators


class Thirring(ModelBase):

    def __init__(self, N: int, g: float, ma: float, lamda: float, s_target: int) -> None:
        """

        Args:
            N: System size.
            g: Bare coupling g.
            ma: Bare mass.
            lamda: Penalty strength (of Lagrangian multiplier).
            s_target: The targeting total Sz charge sector.
        """
        super(Thirring, self).__init__(N)
        self.g = g
        self.ma = ma
        self.lamda = lamda
        self.s_target = s_target

    def _elem(self, site: int) -> np.ndarray:
        Sp, Sm, Sz, I2, O2 = SpinOperators()

        beta = self.g + ((-1.0) ** site * self.ma) - 2.0 * self.lamda * self.s_target
        gamma = self.lamda * (0.25 + self.s_target ** 2 / self.N) + 0.25 * self.g

        return np.array(
            [[I2, -0.5 * Sp, -0.5 * Sm, 2.0 * np.sqrt(self.lamda) * Sz, self.g * Sz, gamma * I2 + beta * Sz],
             [O2, O2, O2, O2, O2, Sm],
             [O2, O2, O2, O2, O2, Sp],
             [O2, O2, O2, I2, O2, np.sqrt(self.lamda) * Sz],
             [O2, O2, O2, O2, O2, Sz],
             [O2, O2, O2, O2, O2, I2]],
            dtype=float
        )
