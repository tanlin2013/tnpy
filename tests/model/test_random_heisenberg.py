import pytest
import numpy as np

from tnpy.model import RandomHeisenberg
from tnpy.operators import FullHamiltonian


class TestRandomHeisenberg:

    @pytest.fixture(scope='class')
    def model(self):
        return RandomHeisenberg(n=6, h=0.5, seed=2022)

    @pytest.fixture(scope='class')
    def offset(self):
        return 0.5

    @pytest.fixture(scope='class')
    def shifted_model(self, model, offset):
        return RandomHeisenberg(
            n=model.n, h=model.h, seed=model.seed, offset=offset
        )

    def test_offset(self, model, offset, shifted_model):
        np.testing.assert_allclose(
            FullHamiltonian(model.mpo).matrix - offset * np.eye(2 ** model.n),
            FullHamiltonian(shifted_model.mpo).matrix,
            atol=1e-12
        )
