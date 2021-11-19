import numpy as np
from tensornetwork import ncon, Tensor
from tnpy.model import ModelBase
from tnpy.operators import SpinOperators


class RandomHeisenberg(ModelBase):

    def __init__(self, N: int, h: float, penalty: float = 0, s_target: int = 0, trial_id: int = None, seed: int = None):
        """
        Args:
            N: System size.
            h: Disorder strength.
            penalty: Penalty strength (of Lagrangian multiplier).
            s_target: The targeting total Sz charge sector.
            trial_id: ID of the current disorder trial.
            seed: Random seed used to initialize the pseudo-random number generator.
        """
        super(RandomHeisenberg, self).__init__(N)
        self.h = h
        self.penalty = penalty
        self.s_target = s_target
        self.trial_id = trial_id
        self._seed = seed
        rng = np.random.RandomState(self.seed)
        self._random_sequence = rng.uniform(-self.h, self.h, size=self.N)

    def _elem(self, site: int) -> np.ndarray:
        Sp, Sm, Sz, I2, O2 = SpinOperators()

        alpha = self.penalty * (0.25 + self.s_target ** 2 / self.N)
        beta = self._random_sequence[site] - 2.0 * self.penalty * self.s_target

        return np.array(
            [[I2, 0.5 * Sp, 0.5 * Sm, 2.0 * self.penalty * Sz, Sz, alpha * I2 + beta * Sz],
             [O2, O2, O2, O2, O2, Sm],
             [O2, O2, O2, O2, O2, Sp],
             [O2, O2, O2, I2, O2, Sz],
             [O2, O2, O2, O2, O2, Sz],
             [O2, O2, O2, O2, O2, I2]],
            dtype=float
        )

    @property
    def seed(self) -> int:
        return self._seed

    @seed.setter
    def seed(self, seed: int) -> None:
        self._seed = seed
        rng = np.random.RandomState(self.seed)
        self._random_sequence = rng.uniform(-self.h, self.h, size=self.N)


class SpectralFoldedRandomHeisenberg(RandomHeisenberg):

    def __init__(self, N: int, h: float, penalty: float = 0, s_target: int = 0, trial_id: int = None, seed: int = None):
        super(SpectralFoldedRandomHeisenberg, self).__init__(N, h, penalty, s_target, trial_id, seed)

    def _elem(self, site: int) -> Tensor:
        M = super()._elem(site)
        return ncon(
            [M, M],
            [(-1, -3, '-a1', 1), (-2, -4, 1, '-b2')]
        ).reshape((36, 36, 2, 2))
