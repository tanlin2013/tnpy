import numpy as np
from collections import namedtuple
from tensornetwork import Node
from typing import Callable


class SpinOperators:

    def __init__(self, spin: float):
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

    @property
    def nodes(self):
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
    def physical_dimensions(self):
        return self._nodes[0].tensor.shape[-1]

    @property
    def bond_dimensions(self):
        return self._nodes[0].tensor.shape[0]
