import time
import logging
import numpy as np
import tensornetwork as tn
from tensornetwork import Node
from tensornetwork.network_operations import conj
from tqdm import tqdm
from itertools import count
from scipy.integrate import solve_ivp
from lib.finite_algorithm_base import FiniteAlgorithmBase
from lib.linalg import svd, qr
from enum import Enum


class Evolve(Enum):
    FORWARD = 1
    BACKWARD = -1


class FiniteTDVP(FiniteAlgorithmBase):

    def __init__(self, mpo, chi, init_method):
        logging.basicConfig(format='%(asctime)s [%(filename)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        logging.root.setLevel(level=logging.INFO)
        super(FiniteTDVP, self).__init__(mpo, chi, init_method)
        self.center_matrices = {}

    def __del__(self):
        pass

    def _unit_solver(self, proceed, t_span, site):

        def forward(t, x):
            M = Node(x.reshape(self.mps_shape(site)))
            W = self.mpo.nodes[site]
            if site == 0:
                Renv = self.right_envs[site]
                Rnorm = self.right_norms[site]
                Renv[0] ^ M[2]
                Renv[1] ^ W[0]
                M[1] ^ W[1]
                Renv[2] ^ Rnorm[0]
                result = Renv @ M @ W @ Rnorm
            elif site == self.N-1:
                Lenv = self.left_envs[site]
                Lnorm = self.left_norms[site]
                Lenv[0] ^ M[0]
                Lenv[1] ^ W[0]
                M[1] ^ W[1]
                Lenv[2] ^ Lnorm[0]
                result = Lenv @ M @ W @ Lnorm
            else:
                Lenv = self.left_envs[site]
                Lnorm = self.left_norms[site]
                Renv = self.right_envs[site]
                Rnorm = self.right_norms[site]
                Lenv[0] ^ M[0]
                Lenv[1] ^ W[0]
                M[1] ^ W[2]
                Renv[0] ^ M[2]
                Renv[1] ^ W[1]
                Lenv[2] ^ Lnorm[0]
                Renv[2] ^ Rnorm[0]
                result = Lenv @ M @ W @ Renv @ Lnorm @ Rnorm
            return -1j * result.tensor.reshape(x.shape)

        def backward(t, x):
            C = Node(x.reshape(self.center_matrices[site].tensor.shape))
            # if site == 0 or site == self.N-1:
            #     raise IndexError("Backward evolution cannot be defined on boundary sites, "
            #                      "solver got site={}".format(site))
            # else:
            Lenv = self.left_envs[site+1]
            Lnorm = self.left_norms[site+1]
            Renv = self.right_envs[site]
            Rnorm = self.right_norms[site]
            Lenv[0] ^ C[0]
            Renv[0] ^ C[1]
            Lenv[1] ^ Renv[1]
            Lenv[2] ^ Lnorm[0]
            Renv[2] ^ Rnorm[0]
            result = Lenv @ C @ Renv @ Lnorm @ Rnorm
            return 1j * result.tensor.reshape(x.shape)

        if proceed == Evolve.FORWARD:
            v0 = self._mps.nodes[site].tensor.reshape(-1).astype(complex)
            result = solve_ivp(forward, t_span, y0=v0)
        elif proceed == Evolve.BACKWARD:
            v0 = self.center_matrices[site].tensor.reshape(-1).astype(complex)
            result = solve_ivp(backward, t_span, y0=v0)
        return result.y[:, -1]

    def sweep(self, iterator, t_span):
        direction = 1 if iterator[0] < iterator[-1] else -1
        for site in iterator:
            theta = self._unit_solver(Evolve.FORWARD, t_span, site)
            if direction == 1:
                theta = theta.reshape(self.d * self.mps_shape(site)[0], -1)
            elif direction == -1:
                theta = theta.reshape(-1, self.d * self.mps_shape(site)[2])
            q, r = qr(theta, chi=self.mps_shape(site)[1+direction])
            print(site, theta.shape, q.shape, r.shape)
            # @TODO: theta.shape is incorrect
            if direction == 1:
                self._mps.nodes[site] = Node(q.reshape(self.mps_shape(site)))
                self.center_matrices[site] = Node(r)
                self._update_left_env(site+1)
                self._update_left_norm(site+1)
                if site < self.N-1:
                    C = Node(self._unit_solver(Evolve.BACKWARD, t_span, site).reshape(r.shape))
                    Mp = self._mps.nodes[site+1]
                    # print(site, C.tensor.shape, Mp.tensor.shape)
                    C[1] ^ Mp[0]
                    self._mps.nodes[site+1] = C @ Mp
            elif direction == -1:
                self._mps.nodes[site] = Node(r.reshape(self.mps_shape(site)))
                self.center_matrices[site-1] = Node(q)
                self._update_right_env(site-1)
                self._update_right_norm(site-1)
                if site > 0:
                    C = Node(self._unit_solver(Evolve.BACKWARD, t_span, site-1).reshape(q.shape))
                    Mp = self._mps.nodes[site-1]
                    Mp[2] ^ C[0]
                    self._mps.nodes[site-1] = Mp @ C
        return

    def evolve(self, t_span):
        pass
