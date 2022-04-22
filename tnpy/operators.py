from __future__ import annotations

from itertools import groupby
from dataclasses import dataclass, InitVar, field, astuple

import numpy as np
import quimb.tensor as qtn


@dataclass
class SpinOperators:
    """
    Constructor of spin operators.

    Attributes:
        Sp: Creation operator, 2 x 2 matrix.
        Sm: Annihilation operator, 2 x 2 matrix.
        Sz: Spin operator on z direction.
        I2: 2 x 2 identity matrix.
        O2: 2 x 2 zero matrix.

    Examples:
            Sp, Sm, Sz, I2, O2 = SpinOperators(spin=0.5)

    Warnings:
        Unpacking ordering is important, and variable-unaware.
    """
    spin: InitVar[float] = field(default=0.5)
    Sp: np.ndarray = field(init=False)
    Sm: np.ndarray = field(init=False)
    Sz: np.ndarray = field(init=False)
    I2: np.ndarray = field(init=False)
    O2: np.ndarray = field(init=False)

    def __post_init__(self, spin: float):
        self.Sp = spin * np.array([[0, 2], [0, 0]], dtype=float)
        self.Sm = spin * np.array([[0, 0], [2, 0]], dtype=float)
        self.Sz = spin * np.array([[1, 0], [0, -1]], dtype=float)
        self.I2 = np.identity(2, dtype=float)
        self.O2 = np.zeros((2, 2), dtype=float)

    def __iter__(self):
        return iter(astuple(self))


class MatrixProductOperator(qtn.MatrixProductOperator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._n_sites = self.nsites
        self._phys_dim = super().phys_dim(0)

        def all_equal(iterable):
            g = groupby(iterable)
            return next(g, True) and not next(g, False)
        assert all_equal(
            [
                super(MatrixProductOperator, self).phys_dim(site)
                for site in range(self.n_sites)
            ]
        )

    @property
    def n_sites(self) -> int:
        return self._n_sites

    @property
    def phys_dim(self) -> int:
        return self._phys_dim

    def square(self) -> MatrixProductOperator:
        """
        Compute the square of :class:`~MatrixProductOperator` (MPO),
        which equivalently merges two MPO layers into one.

        Returns:
            squared_mpo:
        """
        first_layer = self.reindex(
            {f'b{site}': f'dummy{site}' for site in range(self.nsites)}
        )
        second_layer = self.reindex(
            {f'k{site}': f'dummy{site}' for site in range(self.nsites)}
        )
        second_layer.reindex(
            {ind: qtn.rand_uuid() for ind in second_layer.inner_inds()},
            inplace=True
        )

        def _fuse_bilayer(site: int) -> qtn.Tensor:
            bilayer_mpo = first_layer[site] @ second_layer[site]
            if site == 0 or site == self.nsites - 1:
                return bilayer_mpo.fuse(
                    {qtn.rand_uuid(): [
                        bilayer_mpo.inds[0], bilayer_mpo.inds[2]
                    ]}
                )
            return bilayer_mpo.fuse(
                {qtn.rand_uuid(): [bilayer_mpo.inds[0], bilayer_mpo.inds[3]],
                 qtn.rand_uuid(): [bilayer_mpo.inds[1], bilayer_mpo.inds[4]]}
            )
        return MatrixProductOperator(
            [_fuse_bilayer(site).data for site in range(self.nsites)]
        )


class FullHamiltonian:

    def __init__(self, mpo: MatrixProductOperator):
        """
        Construct the Hamiltonian from :class:`~MatrixProductOperator` (MPO).

        Args:
            mpo:

        Examples:
            The matrix element of Hamiltonian can be accessed
            through the property :attr:`~FullHamiltonian.matrix`.

                ham = FullHamiltonian(mpo).matrix

        """
        self._n_sites = mpo.n_sites
        self._phys_dim = mpo.phys_dim

        if self.phys_dim ** self.n_sites > 2 ** 12:
            raise ResourceWarning(
                f"Requesting more than {self.n_sites} sites "
                f"with physical dim {self.phys_dim}."
            )

        self._matrix = mpo.contract().fuse(
            {'0': [f'k{site}' for site in range(self.n_sites)],
             '1': [f'b{site}' for site in range(self.n_sites)]}
        ).data

    @property
    def n_sites(self) -> int:
        return self._n_sites

    @property
    def phys_dim(self) -> int:
        return self._phys_dim

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix
