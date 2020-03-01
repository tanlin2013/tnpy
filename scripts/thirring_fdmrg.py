import os
import sys
sys.path.insert(0, os.path.abspath('..'))
import numpy as np
from lib.operators import SpinOperators, MPO
from lib.finite_dmrg import FiniteDMRG


class Thirring:

    def __init__(self, N, g, ma, lamda, s_target):
        self.N = N
        self.g = g
        self.ma = ma
        self.lamda = lamda
        self.s_target = s_target

    def mpo(self, site):
        Sp, Sm, Sz, I2, O2 = SpinOperators()

        beta = self.g + ((-1.0) ** site * self.ma) - 2.0 * self.lamda * self.s_target
        gamma = self.lamda * (0.25 + self.s_target ** 2 / self.N) + 0.25 * self.g

        return np.array(
            [[I2, -0.5 * Sp, -0.5 * Sm, 2.0 * np.sqrt(self.lamda) * Sz, self.g * Sz, gamma * I2 + beta * Sz],
             [O2, O2, O2, O2, O2, Sm],
             [O2, O2, O2, O2, O2, Sp],
             [O2, O2, O2, I2, O2, np.sqrt(self.lamda) * Sz],
             [O2, O2, O2, O2, O2, Sz],
             [O2, O2, O2, O2, O2, I2]])


if __name__ == "__main__":

    # D = [2, 4, 8, 16, 32, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 32, 16, 8, 4, 2]

    D = [min(2 ** i, 60) for i in range(50 - 1)]
    D += D[::-1]
    N = len(D) + 1
    print(D)

    model = Thirring(N, 1.7, 5.0, 100.0, 0.0)
    fdmrg = FiniteDMRG(D, mpo=MPO(N, model.mpo))
    fdmrg.update(tol=1e-8)
