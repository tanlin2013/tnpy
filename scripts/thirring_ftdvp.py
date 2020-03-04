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

    N = 6
    chi = 10

    model = Thirring(N, g=1.7, ma=5.0, lamda=100.0, s_target=0.0)
    ftdvp = FiniteTDVP(mpo=MPO(N, model.mpo), chi=chi, init_method='random')
    ftdvp._init_norms()
    ftdvp.sweep(np.arange(N), (0, 0.05))
