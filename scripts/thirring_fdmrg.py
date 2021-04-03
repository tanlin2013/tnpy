import numpy as np
from tnpy.operators import SpinOperators, MPO
from tnpy.finite_dmrg import FiniteDMRG


class Thirring:

    def __init__(self, N, g, ma, lamda, s_target) -> None:
        self.N = N
        self.g = g
        self.ma = ma
        self.lamda = lamda
        self.s_target = s_target

    def _elem(self, site: int) -> np.ndarray:
        Sp, Sm, Sz, I2, O2 = SpinOperators()

        beta = self.g + ((-1.0) ** site * self.ma) - 2.0 * self.lamda * self.s_target
        gamma = self.lamda * (0.25 + self.s_target ** 2 / self.N) + 0.25 * self.g

        return np.array(
            [[I2, -0.5 * Sp, -0.5 * Sm, 2.0 * np.sqrt(self.lamda) * Sz, self.g * Sz, gamma * I2 + beta * Sz],
             [O2, O2, O2, O2, O2, Sm],
             [O2, O2, O2, O2, O2, Sp],
             [O2, O2, O2, I2, O2, np.sqrt(self.lamda) * Sz],
             [O2, O2, O2, O2, O2, Sz],
             [O2, O2, O2, O2, O2, I2]],
            dtype=float
        )

    def mpo(self) -> MPO:
        return MPO(self.N, self._elem)


if __name__ == "__main__":

    N = 20
    chi = 60

    model = Thirring(N, g=0.5, ma=1.0, lamda=100.0, s_target=0.0)
    fdmrg = FiniteDMRG(mpo=model.mpo(), chi=chi)
    fdmrg.update(tol=1e-8)
    print(fdmrg.bond_dimensions)
