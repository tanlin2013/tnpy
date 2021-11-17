import time
import logging
import numpy as np
from tensornetwork import Node
from itertools import count
from tnpy.finite_algorithm_base import FiniteAlgorithmBase
from tnpy.linalg import svd, eigshmv
from tnpy.operators import MPO
from typing import Iterable, Union, Tuple


class FiniteDMRG(FiniteAlgorithmBase):

    def __init__(self, mpo: MPO, chi: Union[int, None], init_method='random'):
        """

        Args:
            mpo:
            chi: Maximum bond dimension of MPS
            init_method: 'random' or a filepath
        """
        super(FiniteDMRG, self).__init__(mpo, chi, init_method)

    def _unit_solver(self, site, tol=1e-7) -> Tuple[float, np.ndarray]:
        W = self.mpo[site]

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
        v0 = self._mps.get_tensor(site).reshape(-1, 1)
        return eigshmv(matvec, v0, tol=0.1*tol)

    def _modified_density_matrix(self, site, alpha=0):
        # TODO: return dm
        NotImplemented

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
            logging.info(f"Sweeping to site [{site+1}/{self.N}], E/N = {E/self.N}")
            if direction == 1:
                theta = theta.reshape(self.d * self.mps_shape(site)[0], -1)
            elif direction == -1:
                theta = theta.reshape(-1, self.d * self.mps_shape(site)[2])
            u, s, vt = svd(theta, chi=self.mps_shape(site)[1+direction])
            if direction == 1:
                self._mps.tensors[site] = u.reshape(self.mps_shape(site))
                residual = Node(np.dot(np.diagflat(s), vt))
                M = Node(self._mps.get_tensor(site+1))
                residual[1] ^ M[0]
                self._mps.tensors[site+1] = (residual @ M).tensor
                self._update_left_env(site+1)
            elif direction == -1:
                self._mps.tensors[site] = vt.reshape(self.mps_shape(site))
                residual = Node(np.dot(u, np.diagflat(s)))
                M = Node(self._mps.get_tensor(site-1))
                M[2] ^ residual[0]
                self._mps.tensors[site-1] = (M @ residual).tensor
                self._update_right_env(site-1)
        return E

    def update(self, tol: float = 1e-7, max_sweep: int = 100):
        """

        Args:
            tol:
            max_sweep:

        Returns:

        """
        logging.info(f"Set up tol = {tol}, up to maximally {max_sweep} sweeps")
        clock = [time.process_time()]
        for n_sweep in count(start=1):
            logging.info(f"In sweep epoch [{n_sweep}/{max_sweep}]")
            El = self.sweep(range(self.N-1))
            Er = self.sweep(range(self.N-1, 0, -1))
            clock.append(time.process_time()-clock[-1])
            dE = (El - Er)/self.N
            if abs(dE) < tol:
                break
            elif n_sweep == max_sweep:
                # @TODO: dump mps to file and raise error
                logging.warning(f"Maximum number of sweeps {max_sweep} reached, "
                                f"yet dE/N = {dE} > tol = {tol}")
                break
            elif abs(dE) > tol and dE < 0:
                raise ValueError(f"Fail on lowering energy, got dE/N = {dE}")
        logging.info(f"{n_sweep} loops, best of 3: {np.mean(np.sort(clock)[:3])} sec per loop")


class Projector:

    def __init__(self):
        NotImplemented
