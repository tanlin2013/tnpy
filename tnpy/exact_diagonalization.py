import numpy as np
from tnpy.operators import MatrixProductOperator, FullHamiltonian
from typing import Tuple


class ExactDiagonalization(FullHamiltonian):

    def __init__(self, mpo: MatrixProductOperator):
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

    def reduced_density_matrix(self, site: int, level_idx: int = 0) -> np.ndarray:
        if not 0 <= site < self.n_sites - 1:
            raise ValueError("Parameter `site` for bi-partition has to be within the system size.")
        if not 0 <= level_idx < len(self.evals):
            raise ValueError("Parameter `level_idx` has to be lower than truncation dimension.")
        to_shape = (self.phys_dim ** (site + 1), -1) if site < self.n_sites // 2 \
            else (-1, self.phys_dim ** (self.n_sites - site - 1))
        return np.square(np.linalg.svd(self.evecs[:, level_idx].reshape(to_shape), compute_uv=False))

    def entanglement_entropy(self, site: int, level_idx: int = 0, nan_to_num: bool = False) -> float:
        rho = self.reduced_density_matrix(site, level_idx)
        entropy = -1 * rho @ np.log(rho)
        return np.nan_to_num(entropy) if nan_to_num else entropy
