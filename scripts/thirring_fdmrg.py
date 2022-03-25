from tnpy.model import Thirring
from tnpy.finite_dmrg import FiniteDMRG


if __name__ == "__main__":

    n = 20
    bond_dim = 60

    model = Thirring(n, delta=0.5, ma=1.0, penalty=100.0, s_target=0)
    fdmrg = FiniteDMRG(mpo=model.mpo, bond_dim=bond_dim)
    fdmrg.run(tol=1e-8)
    print(fdmrg.mps)
