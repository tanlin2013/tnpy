import numpy as np
from tnpy.operators import SpinOperators, MPO
from tnpy.tsdrg import TSDRG


class RandomHeisenberg:

    def __init__(self, N: int, h: float, penalty: float = 0, s_target: int = 0, trial_id: int = None):
        """
        Args:
            N: System size.
            h: Disorder strength.
            penalty: Penalty strength (or Lagrangian multiplier).
            s_target: The targeting total Sz charge sector.
            trial_id: ID of the current disorder trial.
        """
        self.N = N
        self.h = h
        self.penalty = penalty
        self.s_target = s_target
        self.trial_id = trial_id

    def _elem(self, site: int) -> np.ndarray:
        Sp, Sm, Sz, I2, O2 = SpinOperators()

        alpha = self.penalty * (0.25 + self.s_target ** 2 / self.N)
        beta = np.random.uniform(-self.h, self.h) - 2.0 * self.penalty * self.s_target

        return np.array(
            [[I2, 0.5 * Sp, 0.5 * Sm, 2.0 * self.penalty * Sz, Sz, alpha * I2 + beta * Sz],
             [O2, O2, O2, O2, O2, Sm],
             [O2, O2, O2, O2, O2, Sp],
             [O2, O2, O2, I2, O2, Sz],
             [O2, O2, O2, O2, O2, Sz],
             [O2, O2, O2, O2, O2, I2]],
            dtype=float
        )

    def mpo(self) -> MPO:
        return MPO(self.N, self._elem)


if __name__ == "__main__":

    N = 10
    h = 3.0
    penalty = 0.0
    s_target = 0
    chi = 32

    model = RandomHeisenberg(N, h, penalty, s_target)
    sdrg = TSDRG(model.mpo(), chi=chi)
    sdrg.run()
    print([tree.id for tree in sdrg.tree])
    # print(sdrg.tree[50])
