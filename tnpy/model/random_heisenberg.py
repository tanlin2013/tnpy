import numpy as np
from tnpy.model import ModelBase
from tnpy.operators import SpinOperators
from .utils import minors_if_no_penalty


class RandomHeisenberg(ModelBase):

    def __init__(self, n: int, h: float, penalty: float = 0, s_target: int = 0,
                 offset: float = 0, trial_id: int = None, seed: int = None):
        """
        Args:
            n: System size.
            h: Disorder strength.
            penalty: Penalty strength (of Lagrangian multiplier).
            s_target: The targeting total Sz charge sector.
            offset: An overall constant to shift the spectrum.
            trial_id: ID of the current disorder trial.
            seed: Random seed used to initialize the pseudo-random number generator.
        """
        super(RandomHeisenberg, self).__init__(n)
        self._h = h
        self._penalty = penalty
        self._s_target = s_target
        self._offset = offset
        self._trial_id = trial_id
        self._seed = seed
        rng = np.random.RandomState(self.seed)
        self._random_sequence = rng.uniform(-self.h, self.h, size=self.n)

    @property
    def h(self) -> float:
        return self._h

    @property
    def penalty(self) -> float:
        return self._penalty

    @property
    def s_target(self) -> int:
        return self._s_target

    @property
    def offset(self) -> float:
        return self._offset

    @property
    def trial_id(self) -> int:
        return self._trial_id

    @minors_if_no_penalty(row=3, col=3)
    def _elem(self, site: int) -> np.ndarray:
        Sp, Sm, Sz, I2, O2 = SpinOperators()

        alpha = self.penalty * (0.25 + self.s_target ** 2 / self.n) + self.offset
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
        self._random_sequence = rng.uniform(-self.h, self.h, size=self.n)
