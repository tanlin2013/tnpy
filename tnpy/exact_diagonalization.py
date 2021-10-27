import numpy as np
from tnpy.operators import MPO, FullHamiltonian
from typing import Tuple


class ExactDiagonalization(FullHamiltonian):

    def __init__(self, mpo: MPO):
        super(ExactDiagonalization, self).__init__(mpo)
        self._evals, self._evecs = self._eigen_solver()

    @property
    def evals(self) -> np.ndarray:
        return self._evals

    @property
    def evecs(self) -> np.ndarray:
        return self._evecs

    def _eigen_solver(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.linalg.eigh(self.matrix)
