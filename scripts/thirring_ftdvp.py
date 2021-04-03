import numpy as np
from tnpy.operators import MPO
from tnpy.finite_tdvp import FiniteTDVP
from thirring_fdmrg import Thirring


if __name__ == "__main__":

    N = 60
    chi = 20

    model = Thirring(N, g=1.7, ma=5.0, lamda=100.0, s_target=0.0)
    ftdvp = FiniteTDVP(mpo=model.mpo(), chi=chi, init_method='random')
    ftdvp._init_norms()
    print(ftdvp.bond_dimensions)
    ftdvp.sweep(np.arange(N-1), (0, 0.05))
