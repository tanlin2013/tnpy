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

    def reduced_density_matrix(self, site: int, energy_level: int = 0) -> np.ndarray:
        to_shape = (self.physical_dimensions ** (site + 1), -1) if site < self.N // 2 \
            else (-1, self.physical_dimensions ** (self.N - site))
        return np.square(np.linalg.svd(self.evecs[:, energy_level].reshape(to_shape), compute_uv=False))

    def entanglement_entropy(self, site: int, energy_level: int = 0) -> float:
        rho = self.reduced_density_matrix(site, energy_level)
        return -1 * rho @ np.log(rho)
