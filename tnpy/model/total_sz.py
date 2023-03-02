import numpy as np

from tnpy.model.model_1d import Model1D
from tnpy.model.utils import boundary_vectors
from tnpy.operators import MatrixProductOperator, SpinOperators


class TotalSz(Model1D):
    def __init__(self, n: int):
        r"""
        .. math::

            S_{tot}^z = \sum_{i=0}^{N-1} S_i^z

        Args:
            n: System size.
        """
        super().__init__(n)

    @boundary_vectors(row=0, col=-1)
    def _elem(self, site: int) -> np.ndarray:
        Sp, Sm, Sz, I2, O2 = SpinOperators()
        return np.array([[I2, Sz], [O2, I2]], dtype=float)

    @boundary_vectors(row=0, col=-1)
    def _rest_elem(self, site: int) -> np.ndarray:
        Sp, Sm, Sz, I2, O2 = SpinOperators()
        return np.array([[I2, O2], [O2, I2]], dtype=float)

    def subsystem_mpo(self, partition_site: int) -> MatrixProductOperator:
        r"""
        Count the total :math:`S^z` magnetization in subsystem A,
        without the other part of system B.
        This can be useful for later calculation on the fluctuation of magnetization.

        Args:
            partition_site: The site to which the system is bipartite into A|B.
                The site itself is included in part A.

        Returns:

        References:
            1. `H. Francis Song, Stephan Rachel, and Karyn Le Hur,
            General relation between entanglement and fluctuations in one dimension,
            Phys. Rev. B 82, 012405 (2010).
            <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.82.012405>`_
        """
        if not 0 <= partition_site < self.n:
            raise ValueError("Partition site must be in between 0 and the system size n.")
        return MatrixProductOperator(
            [
                self._elem(site) if site <= partition_site else self._rest_elem(site)
                for site in range(self.n)
            ]
        )
