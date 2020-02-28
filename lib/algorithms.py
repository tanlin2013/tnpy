import os
import logging
import tqdm
import numpy as np
import tensornetwork as tn
from tensornetwork.network_operations import conj
from tensornetwork.matrixproductstates.finite_mps import FiniteMPS
from itertools import count
from linalg import eigshmv
from operators import MPO, SpinOperators


class FiniteDMRG:

    mps_cls = FiniteMPS

    def __init__(self, D, mpo, init_method='random'):
        logging.basicConfig(format='%(asctime)s [%(filename)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        logging.root.setLevel(level=logging.INFO)
        self.mpo = mpo
        self.d = mpo.physical_dimensions
        self.D = D
        self._mps = None
        self.mps = init_method
        assert(len(D) == self.N-1)
        self.left_envs = []
        self.right_envs = []

    def __del__(self):
        pass

    @property
    def N(self):
        return len(self.mpo.nodes)

    @property
    def dtype(self):
        return self.mpo.nodes[0].tensor.dtype

    @property
    def mps(self):
        return self._mps

    @mps.setter
    def mps(self, init_method='random'):
        if init_method == "random":
            logging.info("Initializing finite MPS randomly")
            self._mps = self.mps_cls.random([self.d]*self.N, self.D, self.dtype)
        elif os.path.isfile(init_method):
            logging.info("Initializing finite MPS from file: {}".format(init_method))
            pass
        else:
            raise KeyError("Invalid init method")
        self._init_envs()

    def _init_envs(self):
        logging.info("Initializing left environments")
        self.left_envs
        logging.info("Initializing right environments")
        self.right_envs

    def _update_left_env(self, site):
        pass

    def _update_right_env(self, site):
        pass

    def _unit_solver(self, site):
        M = self.mpo.nodes[site]
        G_conj = conj(self._mps.nodes[site])
        L = self.left_envs[site]
        R = self.right_envs[site]
        def matvec(x):
            if site == 0:
                pass
            elif site == self.N - 1:
                pass
            else:
                pass
            return result.tensor
        v0 = self._mps.nodes[site].tensor.reshape(-1, 1)
        return eigshmv(matvec, v0)

    def sweep(self, iterator):
        for site in iterator:
            if site == 0:
                pass
            elif site == self.N-1:
                pass
            else:
                pass

    def update(self, tol=1e-7, max_sweep=100):
        logging.info("Set up tol={}, up to maximally '{}' sweeps".format(tol, max_sweep))
        for n_sweep in count():
            pass


if __name__ == "__main__":

    Sp, Sm, Sz, I2, O2 = SpinOperators()

    def th_mpo(site):
        return np.array([[I2, -0.5*Sp, -0.5*Sm, Sz, Sz, I2+Sz],
                        [O2, O2, O2, O2, O2, Sm],
                        [O2, O2, O2, O2, O2, Sp],
                        [O2, O2, O2, I2, O2, Sz],
                        [O2, O2, O2, O2, O2, Sz],
                        [O2, O2, O2, O2, O2, I2]])

    mpo = MPO(5, th_mpo)

    D = [2, 2, 2, 2]
    fdmrg = FiniteDMRG(D, mpo)

    t = fdmrg.mps.nodes[0]
    t2 = conj(t)
    t[1] ^ t2[1]
    t[0] ^ t2[0]
    result = t @ t2
    print(result)
