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
from scipy.integrate import solve_ivp
from lib.linalg import svd
from lib.operators import MPO, SpinOperators


class FiniteTDVP:

    def __init__(self, mpo):
        self.mpo = mpo

    def _update_left_env(self, site):
        W = self.mpo.nodes[site - 1]
        M = self._mps.nodes[site - 1]
        M_conj = conj(self._mps.nodes[site - 1])
        if site == 1:
            M[0] ^ M_conj[0]
            M[1] ^ W[1]
            W[2] ^ M_conj[1]
            self.left_envs[site] = M @ W @ M_conj
        else:
            L = self.left_envs[site - 1]
            L[0] ^ M[0]
            L[1] ^ W[0]
            L[2] ^ M_conj[0]
            M[1] ^ W[2]
            M_conj[1] ^ W[3]
            self.left_envs[site] = L @ M @ W @ M_conj

    def _update_right_env(self, site):
        W = self.mpo.nodes[site + 1]
        M = self._mps.nodes[site + 1]
        M_conj = conj(self._mps.nodes[site + 1])
        if site == self.N - 2:
            M[2] ^ M_conj[2]
            M[1] ^ W[1]
            W[2] ^ M_conj[1]
            self.right_envs[site] = M @ W @ M_conj
        else:
            R = self.right_envs[site + 1]
            R[0] ^ M[2]
            R[1] ^ W[1]
            R[2] ^ M_conj[2]
            M[1] ^ W[2]
            M_conj[1] ^ W[3]
            self.right_envs[site] = R @ M @ W @ M_conj

    def _update_left_norm(self, site):
        pass

    def _update_right_norm(self, site):
        pass

    def _unit_solver(self, site):

        def projected_ham(x):
            pass
        return solve_ivp(projected_ham, )

    def sweep(self):
        pass

    def update(self):
        pass