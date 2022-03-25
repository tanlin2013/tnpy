import numpy as np
from tnpy.model import Thirring
from tnpy.finite_tdvp import FiniteTDVP


if __name__ == "__main__":

    n = 60
    bond_dim = 20

    model = Thirring(n, delta=1.7, ma=5.0, penalty=100.0, s_target=0)
    ftdvp = FiniteTDVP(mpo=model.mpo, chi=bond_dim, init_method='random')
    ftdvp._init_norms()
    ftdvp.sweep(np.arange(n - 1), (0, 0.05))
