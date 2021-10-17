import numpy as np
from collections import namedtuple
from tensornetwork import Node
from typing import Callable, List


class SpinOperators:

    def __init__(self, spin: float = 0.5):
        self.spin = spin

    def __new__(cls, spin: float = 0.5):
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

    @property
    def nodes(self) -> List[Node]:
        return self._nodes

    @nodes.setter
    def nodes(self, func: Callable):
        for site in range(self.N):
            if site == 0:
                self._nodes.append(Node(func(site)[0, :, :, :]))
            elif site == self.N-1:
                self._nodes.append(Node(func(site)[:, -1, :, :]))
            else:
                self._nodes.append(Node(func(site)))

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
