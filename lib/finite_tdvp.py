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

    def __init__(self, mpo):
        logging.basicConfig(format='%(asctime)s [%(filename)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        logging.root.setLevel(level=logging.INFO)
        super(FiniteTDVP, self).__init__(D, mpo, init_method)

    def __del__(self):
        pass

    def _unit_solver(self, site):

        def projected_ham(x):
            pass
        return solve_ivp(projected_ham, )

    def sweep(self):
        pass

    def update(self):
        pass
