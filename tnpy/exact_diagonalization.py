from typing import Tuple, Sequence

import numpy as np

from tnpy import logger
from tnpy.operators import MatrixProductOperator, FullHamiltonian


class ExactDiagonalization(FullHamiltonian):
    def __init__(self, mpo: MatrixProductOperator):
        """
        Perform the numerically exact diagonalization on the matrix,
        which is constructed through the given Matrix Product Operator (MPO).
        Calculations are taken in prompt on the initialization of this class.

        Args:
            mpo: The matrix product operator.

        Raises:
            ResourceWarning: When the dimensions of Hamiltonian are larger
                than :math:`4096 \times 4096`.

        Examples:
            >>> ed = ExactDiagonalization(mpo)
            >>> evals = ed.evals
            >>> evecs = ed.evecs
        """
        super().__init__(mpo)
        self._evals, self._evecs = self._eigen_solver()

    @property
    def evals(self) -> np.ndarray:
        """
        Eigenvalues in ascending order.

        Returns:

        """
        return self._evals

    @property
    def evecs(self) -> np.ndarray:
        """
        Eigenvectors in the order accordingly to :attr:`~ExactDiagonalization.evals`.

        Returns:
            The eigenvectors,
            with the column ``v[:, k]`` is the eigenvector corresponding to
            the k-th eigenvalue ``w[k]``.
        """
        return self._evecs

    def _eigen_solver(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.linalg.eigh(self.matrix)

    def reduced_density_matrix(self, site: int, level_idx: int = 0) -> np.ndarray:
        if not 0 <= site < self.n_sites - 1:
            raise ValueError(
                "Parameter `site` for bi-partition has to be within the system size."
            )
        if not 0 <= level_idx < len(self.evals):
            raise ValueError(
                "Parameter `level_idx` has to be lower than truncation dimension."
            )
        to_shape = (
            (self.phys_dim ** (site + 1), -1)
            if site < self.n_sites // 2
            else (-1, self.phys_dim ** (self.n_sites - site - 1))
        )
        return np.square(
            np.linalg.svd(self.evecs[:, level_idx].reshape(to_shape), compute_uv=False)
        )

    def entanglement_entropy(
        self, site: int, level_idx: int = 0, nan_to_num: bool = False
    ) -> float:
        rho = self.reduced_density_matrix(site, level_idx)
        entropy = -1 * rho @ np.log(rho)
        return np.nan_to_num(entropy) if nan_to_num else entropy

    @staticmethod
    def kron_operators(operators: Sequence[np.ndarray]) -> np.ndarray:
        """
        Perform Kronecker product on the given sequence of operators.

        Args:
            operators: A list of operators to take Kronecker product.

        Returns:
            The resulting product operator.
        """
        opt = operators[0]
        for next_opt in operators[1:]:
            opt = np.kron(opt, next_opt)
        return opt

    def one_point_function(
        self, operator: np.ndarray, site: int, level_idx: int = 0
    ) -> float:
        assert operator.shape == (self.phys_dim, self.phys_dim)
        opt_mat = self.kron_operators(
            [
                np.eye(self.phys_dim**site),
                operator,
                np.eye(self.phys_dim ** (self.n_sites - site - 1)),
            ]
        )
        return self.evecs[:, level_idx].T @ opt_mat @ self.evecs[:, level_idx]

    def two_point_function(
        self,
        operator1: np.ndarray,
        operator2: np.ndarray,
        site1: int,
        site2: int,
        level_idx: int,
    ) -> float:
        assert operator1.shape == (self.phys_dim, self.phys_dim)
        assert operator2.shape == (self.phys_dim, self.phys_dim)
        assert site1 != site2
        site1, site2 = sorted((site1, site2))
        # TODO: operators aren't sorted accordingly
        opt_mat = self.kron_operators(
            [
                np.eye(self.phys_dim**site1),
                operator1,
                np.eye(self.phys_dim ** (abs(site1 - site2) - 1)),
                operator2,
                np.eye(self.phys_dim ** (self.n_sites - site2 - 1)),
            ]
        )
        return self.evecs[:, level_idx].T @ opt_mat @ self.evecs[:, level_idx]

    def connected_two_point_function(
        self,
        operator1: np.ndarray,
        operator2: np.ndarray,
        site1: int,
        site2: int,
        level_idx: int,
    ) -> float:
        return self.two_point_function(
            operator1, operator2, site1, site2, level_idx
        ) - self.one_point_function(
            operator1, site1, level_idx
        ) * self.one_point_function(
            operator2, site2, level_idx
        )

    def variance(self, operator: np.ndarray = None, tol: float = 1e-12) -> np.ndarray:
        """
        Compute the variance on input operator.

        Args:
            operator: Default None to the Hamailtonian itself.
            tol: The numerical tolerance.

        Returns:
            The variance.

        Raises:
            Warnings: If any off-diagonal element is larger than ``tol``.
        """
        operator = self.matrix if operator is None else operator
        var = (
            self.evecs.T @ np.linalg.matrix_power(operator, 2) @ self.evecs
            - (self.evecs.T @ operator @ self.evecs) ** 2
        )
        if not np.allclose(
            np.zeros(var.shape),
            var - np.diagflat(np.diag(var)),
            atol=tol,
        ):
            logger.warning("Expectation value may contain large off-diagonal elements.")
        return np.diag(var)
