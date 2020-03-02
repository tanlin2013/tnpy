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
from lib.linalg import svd
from lib.operators import MPO


class FiniteTDVP(FiniteAlgorithmBase):

    def __init__(self, mpo, init_method):
        logging.basicConfig(format='%(asctime)s [%(filename)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        logging.root.setLevel(level=logging.INFO)
        super(FiniteTDVP, self).__init__(D, mpo, init_method)

    def __del__(self):
        pass

    def _unit_solver(self, proceed, site):

        def forward(x):
            M = Node(x.reshape(self.mps_shape(site)))
            W = self.mpo.nodes[site]
            if site == 0:
                pass
            elif site == self.N-1:
                pass
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
            return result.tensor.reshape(x.shape)

        def backward(x):
            M = Node(x.reshape(self.mps_shape(site)))
            if site == 0:
                pass
            elif site == self.N-1:
                pass
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
            return result.tensor.reshape(x.shape)

        if proceed == 1:
            return solve_ivp(forward, t_span=, y0=)
        elif proceed == -1:
            return solve_ivp(backward, t_span=, y0=)

    def sweep(self, iterator, delta):
        direction = 1 if iterator[0] < iterator[-1] else -1
        for site in iterator:
            M = self._unit_solver(site)

    def evolve(self, t_span):
        pass
