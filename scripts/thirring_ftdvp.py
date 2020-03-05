import numpy as np
# try:
#     from TNpy.operators import SpinOperators, MPO
#     from TNpy.finite_dmrg import FiniteDMRG
# except ImportError:
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
from thirring_fdmrg import Thirring
from lib.operators import MPO
from lib.finite_tdvp import FiniteTDVP


if __name__ == "__main__":

    N = 60
    chi = 20

    model = Thirring(N, g=1.7, ma=5.0, lamda=100.0, s_target=0.0)
    ftdvp = FiniteTDVP(mpo=MPO(N, model.mpo), chi=chi, init_method='random')
    ftdvp._init_norms()
    print(ftdvp.bond_dimensions)
    ftdvp.sweep(np.arange(N-1), (0, 0.05))
