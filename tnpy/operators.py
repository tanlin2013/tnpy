import numpy as np
import quimb.tensor as qtn
from tensornetwork import Node
from collections import namedtuple
from itertools import groupby
from typing import Callable, List
import warnings


class SpinOperators:

    def __new__(cls, spin: float = 0.5) -> namedtuple:
        super(SpinOperators, cls).__init__(spin)
        SOp = namedtuple('SpinOperators', ['Sp', 'Sm', 'Sz', 'I2', 'O2'])
        return SOp(Sp=spin * np.array([[0, 2], [0, 0]], dtype=float),
                   Sm=spin * np.array([[0, 0], [2, 0]], dtype=float),
                   Sz=spin * np.array([[1, 0], [0, -1]], dtype=float),
                   I2=np.identity(2, dtype=float),
                   O2=np.zeros((2, 2), dtype=float))


class MPO:

    def __init__(self, N: int, func: Callable):
        warnings.warn("Deprecated, pls use tnpy.operators.MatrixProductOperator instead.", DeprecationWarning)
        self.N = N
        self._nodes = []
        self.nodes = func
        self.__identity = np.identity(self.bond_dimensions)
        self._v_left = Node(self.__identity[0])
        self._v_right = Node(self.__identity[-1])

    def __getitem__(self, site: int) -> Node:
        return self._nodes[site]

    @property
    def nodes(self) -> List[Node]:
        return self._nodes

    @nodes.setter
    def nodes(self, func: Callable):
        def with_open_boundary(site: int) -> np.ndarray:
            if site == 0:
                return func(site)[0, :, :, :]
            elif site == self.N - 1:
                return func(site)[:, -1, :, :]
            return func(site)
        self._nodes = [Node(with_open_boundary(site)) for site in range(self.N)]

    @property
    def physical_dimensions(self) -> int:
        return self._nodes[0].tensor.shape[-1]

    @property
    def bond_dimensions(self) -> int:
        return self._nodes[0].tensor.shape[0]

    @property
    def v_left(self) -> Node:
        return self._v_left

    @v_left.setter
    def v_left(self, header: int):
        assert header in [0, -1], "header can only accept 0 or -1"
        self._v_left = Node(self.__identity[header])

    @property
    def v_right(self) -> Node:
        return self._v_right

    @v_right.setter
    def v_right(self, header: int):
        assert header in [0, -1], "header can only accept 0 or -1"
        self._v_right = Node(self.__identity[header])


class MatrixProductOperator(qtn.MatrixProductOperator):

    def __init__(self, *args, **kwargs):
        super(MatrixProductOperator, self).__init__(*args, **kwargs)

    def square(self) -> qtn.MatrixProductOperator:
        first_layer = self.reindex(
            {f'b{site}': f'dummy{site}' for site in range(self.nsites)}
        )
        second_layer = self.reindex(
            {f'k{site}': f'dummy{site}' for site in range(self.nsites)}
        )
        second_layer.reindex(
            {ind: qtn.rand_uuid() for ind in second_layer.inner_inds()}, inplace=True
        )

        def _fuse_bilayer(site: int) -> qtn.Tensor:
            bilayer_mpo = first_layer[site] @ second_layer[site]
            if site == 0 or site == self.nsites - 1:
                return bilayer_mpo.fuse({qtn.rand_uuid(): [bilayer_mpo.inds[0], bilayer_mpo.inds[2]]})
            return bilayer_mpo.fuse(
                {qtn.rand_uuid(): [bilayer_mpo.inds[0], bilayer_mpo.inds[3]],
                 qtn.rand_uuid(): [bilayer_mpo.inds[1], bilayer_mpo.inds[4]]}
            )
        return qtn.MatrixProductOperator([_fuse_bilayer(site).data for site in range(self.nsites)])


class FullHamiltonian:

    def __init__(self, mpo: MatrixProductOperator):
        self._n_sites = mpo.nsites
        self._phys_dim = mpo.phys_dim(0)

        def all_equal(iterable):
            g = groupby(iterable)
            return next(g, True) and not next(g, False)
        assert all_equal([mpo.phys_dim(site) for site in range(self.n_sites)])

        if self.phys_dim ** self.n_sites > 2 ** 12:
            raise ResourceWarning(f"Requesting more than {self.n_sites} sites with physical dim {self.phys_dim}.")

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
