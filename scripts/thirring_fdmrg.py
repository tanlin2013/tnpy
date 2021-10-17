from tnpy.model import Thirring
from tnpy.finite_dmrg import FiniteDMRG


if __name__ == "__main__":

    N = 20
    chi = 60

    model = Thirring(N, g=0.5, ma=1.0, lamda=100.0, s_target=0)
    fdmrg = FiniteDMRG(mpo=model.mpo, chi=chi)
    fdmrg.update(tol=1e-8)
    print(fdmrg.bond_dimensions)
