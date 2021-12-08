import numpy as np
from collections import namedtuple
from tensornetwork import Node, Tensor, ncon
from typing import Callable, List, Union, Tuple


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


class FullHamiltonian:

    def __init__(self, mpo: MPO):
        self._N = mpo.N
        self._matrix = None
        self.matrix = mpo
        self._physical_dimensions = mpo.physical_dimensions

    @property
    def N(self) -> int:
        return self._N

    @property
    def physical_dimensions(self) -> int:
        return self._physical_dimensions

    @property
    def matrix(self) -> Union[np.ndarray, Tensor]:
        return self._matrix

    @matrix.setter
    def matrix(self, mpo: MPO):
        def network_structure(site: int) -> Union[Tuple[int, str, str], Tuple[int, int, str, str]]:
            if site == 1:
                return site, f'-a{site}', f'-b{site}'
            elif site == self.N:
                return site-1, f'-a{site}', f'-b{site}'
            return site-1, site, f'-a{site}', f'-b{site}'
        self._matrix = ncon(
            [node.tensor for node in mpo],
            [network_structure(site) for site in range(1, self.N + 1)],
            out_order=[f'-a{site}' for site in range(1, self.N + 1)] +
                      [f'-b{site}' for site in range(1, self.N + 1)]
        )
        self._matrix = self._matrix.reshape(
            (
                mpo.physical_dimensions ** self.N,
                mpo.physical_dimensions ** self.N
            )
        )
