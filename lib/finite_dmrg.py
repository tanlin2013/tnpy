import time
import logging
import numpy as np
import tensornetwork as tn
from tensornetwork import Node
from tqdm import tqdm
from itertools import count
from lib.finite_algorithm_base import FiniteAlgorithmBase
from lib.linalg import svd, eigshmv
from lib.operators import MPO
from typing import List, Iterable


class FiniteDMRG(FiniteAlgorithmBase):

    def __init__(self, D: List[int], mpo: MPO, init_method='random'):
        """

        Args:
            D:
            mpo:
            init_method:
        """
        logging.basicConfig(format='%(asctime)s [%(filename)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        logging.root.setLevel(level=logging.INFO)
        super(FiniteDMRG, self).__init__(D, mpo, init_method)

    def __del__(self):
        pass

    def _init_envs(self):
        # @TODO: only need to do for one direction
        logging.info("Initializing left environments")
        for site in tqdm(range(1, self.N)):
            self._update_left_env(site)
        logging.info("Initializing right environments")
        for site in tqdm(range(self.N-2, -1, -1)):
            self._update_right_env(site)

    def _unit_solver(self, site, tol=1e-7):
        W = self.mpo.nodes[site]

        def matvec(x):
            M = Node(x.reshape(self.mps_shape(site)))
            if site == 0:
                R = self.right_envs[site]
                R[0] ^ M[2]
                R[1] ^ W[0]
                M[1] ^ W[1]
                result = M @ W @ R
            elif site == self.N-1:
                L = self.left_envs[site]
                L[0] ^ M[0]
                L[1] ^ W[0]
                M[1] ^ W[1]
                result = L @ M @ W
            else:
                L = self.left_envs[site]
                R = self.right_envs[site]
                L[0] ^ M[0]
                L[1] ^ W[0]
                R[0] ^ M[2]
                R[1] ^ W[1]
                M[1] ^ W[2]
                result = L @ M @ W @ R
            return result.tensor.reshape(x.shape)
        v0 = self._mps.nodes[site].tensor.reshape(-1, 1)
        return eigshmv(matvec, v0, tol=0.1*tol)

    def _modified_density_matrix(self, site, alpha=0):
        # @TODO: Not Implemented
        # return dm
        pass

    def sweep(self, iterator: Iterable, tol: float = 1e-7) -> float:
        """

        Args:
            iterator:
            tol:

        Returns:

        """
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
                M = self._mps.nodes[site+1]
                residual[1] ^ M[0]
                self._mps.nodes[site+1] = residual @ M
                self._update_left_env(site+1)
            elif direction == -1:
                self._mps.nodes[site] = Node(vt.reshape(self.mps_shape(site)))
                residual = Node(np.dot(u, np.diagflat(s)))
                M = self._mps.nodes[site-1]
                M[2] ^ residual[0]
                self._mps.nodes[site-1] = M @ residual
                self._update_right_env(site-1)
        return E

    def update(self, tol: float = 1e-7, max_sweep: int = 100):
        """

        Args:
            tol:
            max_sweep:

        Returns:

        """
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
                # @TODO: dump mps to file and raise error
                logging.warning("Maximum number of sweeps {} reached, "
                                "yet dE/N = {} > tol = {}".format(max_sweep, dE, tol))
                break
            elif abs(dE) > tol and dE < 0:
                raise ValueError("Fail on lowering energy, got dE/N = {}".format(dE))
        logging.info("{} loops, best of 3: {} sec per loop"
                     "".format(n_sweep, np.mean(np.sort(clock)[:3])))
