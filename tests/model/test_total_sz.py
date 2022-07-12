import pytest

import numpy as np

from tnpy.model import TotalSz
from tnpy.operators import FullHamiltonian


@pytest.mark.parametrize("n, site", [(6, 3), (8, 4), (10, 7)])
def test_subsystem_mpo(n, site):
    np.testing.assert_array_equal(
        np.kron(
            FullHamiltonian(TotalSz(n=site + 1).subsystem_mpo(site)).matrix,
            np.eye(2 ** (n - site - 1)),
        ),
        FullHamiltonian(TotalSz(n=n).subsystem_mpo(site)).matrix,
    )
    np.testing.assert_array_equal(
        np.kron(
            FullHamiltonian(TotalSz(n=site + 2).subsystem_mpo(site)).matrix,
            np.eye(2 ** (n - site - 2)),
        ),
        FullHamiltonian(TotalSz(n=n).subsystem_mpo(site)).matrix,
    )
