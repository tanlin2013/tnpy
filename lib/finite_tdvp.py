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

    def __del__(self):
        pass

    def _unit_solver(self, proceed, t_span, site):

        def forward(x):
            M = Node(x.reshape(self.mps_shape(site)))
            W = self.mpo.nodes[site]
            if site == 0:
                Renv = self.right_envs[site]
                Rnorm = self.right_norms[site]
                Renv[0] ^ M[2]
                Renv[1] ^ W[0]
                Renv[2] ^ Rnorm[0]
                result = Renv @ M @ W @ Rnorm
            elif site == self.N-1:
                Lenv = self.left_envs[site]
                Lnorm = self.left_norms[site]
                Lenv[0] ^ M[0]
                Lenv[1] ^ W[0]
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

        def backward(x):
            M = Node(x.reshape(self.mps_shape(site)))
            if site == 0 or site == self.N-1:
                raise IndexError("Backward evolution cannot be defined on boundary sites, "
                                 "solver got site={}".format(site))
            else:
                Lenv = self.left_envs[site]
                Lnorm = self.left_norms[site]
                Renv = self.right_envs[site]
                Rnorm = self.right_norms[site]
                Lenv[0] ^ M[0]
                Renv[0] ^ M[2]
                Lenv[1] ^ Renv[1]
                Lenv[2] ^ Lnorm[0]
                Renv[2] ^ Rnorm[0]
                result = Lenv @ M @ Renv @ Lnorm @ Rnorm
            return 1j * result.tensor.reshape(x.shape)

        v0 = self._mps.nodes[site].tensor.reshape(-1, 1)
        if proceed == Evolve.FORWARD:
            return solve_ivp(forward, t_span, y0=v0)
        elif proceed == Evolve.BACKWARD:
            return solve_ivp(backward, t_span, y0=v0)

    def sweep(self, iterator, t_span):
        direction = 1 if iterator[0] < iterator[-1] else -1
        for site in iterator:
            theta = self._unit_solver(Evolve.FORWARD, t_span, site)
            q, r = qr(theta, )

    def evolve(self, t_span):
        pass
