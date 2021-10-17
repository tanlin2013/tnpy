import numpy as np
from tnpy.model import ModelBase
from tnpy.operators import SpinOperators


class DimerXXZ(ModelBase):

    def __init__(self, N: int, J: float, delta: float, h: float,
                 penalty: float = 0, s_target: int = 0, trial_id: int = None):
        """
        Args:
            N: System size.
            J:
            delta:
            h: Disorder strength.
            penalty: Penalty strength (or Lagrangian multiplier).
            s_target: The targeting total Sz charge sector.
            trial_id: ID of the current disorder trial.
        """
        super(DimerXXZ, self).__init__(N)
        self.J = J
        self.delta = delta
        self.h = h
        self.penalty = penalty
        self.s_target = s_target
        self.trial_id = trial_id

    def _elem(self, site: int) -> np.ndarray:
        Sp, Sm, Sz, I2, O2 = SpinOperators(spin=1)

        rand_J = (1 + self.delta * (-1) ** site) * np.random.uniform() ** self.J
        alpha = self.penalty * (0.25 + self.s_target ** 2 / self.N)
        beta = np.random.uniform(-self.h, self.h) - 2.0 * self.penalty * self.s_target

        return np.array(
            [[I2, 0.5 * rand_J * Sp, 0.5 * rand_J * Sm, 2.0 * self.penalty * Sz, Sz, alpha * I2 + beta * Sz],
             [O2, O2, O2, O2, O2, Sm],
             [O2, O2, O2, O2, O2, Sp],
             [O2, O2, O2, I2, O2, Sz],
             [O2, O2, O2, O2, O2, Sz],
             [O2, O2, O2, O2, O2, I2]],
            dtype=float
        )
