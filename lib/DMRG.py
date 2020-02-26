import logging
import numpy as np
import tensornetwork as tn
# from tensornetwork.matrixproductstates.infinite_mps import InfiniteMPS
from tensornetwork.matrixproductstates.finite_mps import FiniteMPS


class FiniteDMRG:

    fmps_cls = FiniteMPS

    def __init__(self, d, D, init_method='random', dtype=np.float):
        self._d = d
        self._D = D
        self.dtype = dtype
        self._fmps = None
        self.fmps = init_method

    def __del__(self):
        pass

    @property
    def d(self):
        return self._d

    @property
    def D(self):
        return self._D

    @property
    def fmps(self):
        return self._fmps

    @fmps.setter
    def fmps(self, init_method='random'):
        if init_method == "random":
            self._fmps = self.fmps_cls.random(self.d, self.D, self.dtype)
        elif init_method == 'from_file':
            pass
        for node in self._fmps.nodes:
            print(node.tensor.shape)
            print(node.tensor)

    def update(self):
        pass


if __name__ == "__main__":

    d = [2, 2, 2]
    D = [2, 2]

    agent = FiniteDMRG(d, D)
