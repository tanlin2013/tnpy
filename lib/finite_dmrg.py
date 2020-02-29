import os
import time
import logging
import numpy as np
import tensornetwork as tn
from tensornetwork import Node
from tensornetwork.network_operations import conj
from tensornetwork.matrixproductstates.finite_mps import FiniteMPS
from tqdm import tqdm
from itertools import count
from lib.linalg import svd, eigshmv
from lib.operators import MPO, SpinOperators


class FiniteDMRG:

    mps_cls = FiniteMPS

    def __init__(self, D, mpo, init_method='random'):
        logging.basicConfig(format='%(asctime)s [%(filename)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        logging.root.setLevel(level=logging.INFO)
        self.mpo = mpo
        self.d = mpo.physical_dimensions
        self.D = D
        self.left_envs = {}
        self.right_envs = {}
        self._mps = None
        self.reset_mps = init_method
        assert(len(D) == self.N-1)

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
    def reset_mps(self, init_method='random'):
        if init_method == "random":
            logging.info("Initializing finite MPS randomly")
            self._mps = self.mps_cls.random([self.d]*self.N, self.D, self.dtype)
            # self.normalize_mps(direction=1, normalize_sv=True)
        elif os.path.isfile(init_method):
            logging.info("Initializing finite MPS from file: {}".format(init_method))
            # @TODO: Not Implemented
            pass
        else:
            raise KeyError("Invalid init method")
        self._init_envs()

    def mps_shape(self, site):
        return self._mps.nodes[site].tensor.shape

    # def normalize_mps(self, direction, normalize_sv=False):
    #     if direction == 1:
    #         iterator = range(self.N-1)
    #     elif direction == -1:
    #         iterator = range(self.N-1, 0, -1)
    #
    #     for site in iterator:
    #         theta = self._mps.nodes[site].tensor
    #         if direction == 1:
    #             theta = theta.reshape(self.d * self.mps_shape(site)[0], -1)
    #         elif direction == -1:
    #             theta = theta.reshape(-1, self.d * self.mps_shape(site)[2])
    #         u, s, vt = np.linalg.svd(theta, full_matrices=False)
    #         if normalize_sv:
    #             s /= np.linalg.norm(s)
    #         if direction == 1:
    #             self._mps.nodes[site] = Node(u.reshape(self.mps_shape(site)))
    #             residual = Node(np.dot(np.diagflat(s), vt))
    #             G = self._mps.nodes[site+1]
    #             residual[1] ^ G[0]
    #             self._mps.nodes[site+1] = residual @ G
    #             self._update_left_env(site+1)
    #         elif direction == -1:
    #             self._mps.nodes[site] = Node(vt.reshape(self.mps_shape(site)))
    #             residual = Node(np.dot(u, np.diagflat(s)))
    #             G = self._mps.nodes[site-1]
    #             G[2] ^ residual[0]
    #             self._mps.nodes[site-1] = G @ residual
    #             self._update_right_env(site-1)

    def _init_envs(self):
        logging.info("Initializing left environments")
        for site in tqdm(range(1, self.N)):
            self._update_left_env(site)
        logging.info("Initializing right environments")
        for site in tqdm(range(self.N-2, -1, -1)):
            self._update_right_env(site)

    def _update_left_env(self, site):
        M = self.mpo.nodes[site-1]
        G = self._mps.nodes[site-1]
        G_conj = conj(self._mps.nodes[site-1])
        if site == 1:
            G[0] ^ G_conj[0]
            G[1] ^ M[1]
            M[2] ^ G_conj[1]
            self.left_envs[site] = G @ M @ G_conj
        else:
            L = self.left_envs[site-1]
            L[0] ^ G[0]
            L[1] ^ M[0]
            L[2] ^ G_conj[0]
            G[1] ^ M[2]
            G_conj[1] ^ M[3]
            self.left_envs[site] = L @ G @ M @ G_conj

    def _update_right_env(self, site):
        M = self.mpo.nodes[site+1]
        G = self._mps.nodes[site+1]
        G_conj = conj(self._mps.nodes[site+1])
        if site == self.N-2:
            G[2] ^ G_conj[2]
            G[1] ^ M[1]
            M[2] ^ G_conj[1]
            self.right_envs[site] = G @ M @ G_conj
        else:
            R = self.right_envs[site+1]
            R[0] ^ G[2]
            R[1] ^ M[1]
            R[2] ^ G_conj[2]
            G[1] ^ M[2]
            G_conj[1] ^ M[3]
            self.right_envs[site] = R @ G @ M @ G_conj

    def _unit_solver(self, site, tol=1e-7):
        M = self.mpo.nodes[site]

        def matvec(x):
            G = Node(x.reshape(self.mps_shape(site)))
            if site == 0:
                R = self.right_envs[site]
                R[0] ^ G[2]
                R[1] ^ M[0]
                G[1] ^ M[1]
                result = G @ M @ R
            elif site == self.N-1:
                L = self.left_envs[site]
                L[0] ^ G[0]
                L[1] ^ M[0]
                G[1] ^ M[1]
                result = L @ G @ M
            else:
                L = self.left_envs[site]
                R = self.right_envs[site]
                L[0] ^ G[0]
                L[1] ^ M[0]
                R[0] ^ G[2]
                R[1] ^ M[1]
                G[1] ^ M[2]
                result = L @ G @ M @ R
            return result.tensor.reshape(x.shape)
        v0 = self._mps.nodes[site].tensor.reshape(-1, 1)
        return eigshmv(matvec, v0, tol=0.1*tol)

    def _modified_density_matrix(self, site, alpha=0):
        # return dm
        pass

    def sweep(self, iterator, tol=1e-7):
        direction = 1 if iterator[0] < iterator[-1] else -1
        for site in iterator:
            E, theta = self._unit_solver(site, tol)
            logging.info("Sweeping to site [{}/{}], E/N = {}".format(site+1, self.N, E/self.N))
            if direction == 1:
                theta = theta.reshape(self.d * self.mps_shape(site)[0], -1)
            elif direction == -1:
                theta = theta.reshape(-1, self.d * self.mps_shape(site)[2])
            u, s, vt = svd(theta, chi=self.D[site+min(0, direction)])
            if direction == 1:
                self._mps.nodes[site] = Node(u.reshape(self.mps_shape(site)))
                residual = Node(np.dot(np.diagflat(s), vt))
                G = self._mps.nodes[site+1]
                residual[1] ^ G[0]
                self._mps.nodes[site+1] = residual @ G
                self._update_left_env(site+1)
            elif direction == -1:
                self._mps.nodes[site] = Node(vt.reshape(self.mps_shape(site)))
                residual = Node(np.dot(u, np.diagflat(s)))
                G = self._mps.nodes[site-1]
                G[2] ^ residual[0]
                self._mps.nodes[site-1] = G @ residual
                self._update_right_env(site-1)
        return E

    def update(self, tol=1e-7, max_sweep=100):
        logging.info("Set up tol = {}, up to maximally {} sweeps".format(tol, max_sweep))
        clock = [time.process_time()]
        for n_sweep in count(start=1):
            logging.info("In sweep epoch [{}/{}]".format(n_sweep, max_sweep))
            El = self.sweep(range(self.N-1))
            Er = self.sweep(range(self.N-1, 0, -1))
            clock.append(time.process_time()-clock[-1])
            dE = (El - Er)/self.N
            if abs(dE) < tol:
                break
            elif n_sweep == max_sweep:
                logging.warning("Maximum number of sweeps {} reached, "
                                "yet dE/N = {} > tol = {}".format(max_sweep, dE, tol))
                break
            elif abs(dE) > tol and dE < 0:
                raise ValueError("Fail on lowering energy, got dE/N = {}".format(dE))
        logging.info("{} loops, best of 3: {} sec per loop"
                     "".format(n_sweep, np.mean(np.sort(clock)[:3])))

    def save_mps(self):
        pass
