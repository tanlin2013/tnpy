from unittest import TestCase
import numpy as np
from tnpy.model import RandomHeisenberg
from tnpy.operators import FullHamiltonian


class TestRandomHeisenberg(TestCase):

    def test_offset(self):
        model = RandomHeisenberg(n=6, h=0.5, seed=2022)
        offset = 0.5
        shifted_model = RandomHeisenberg(
            n=model.n, h=model.h, seed=model.seed, offset=offset
        )
        np.testing.assert_allclose(
            FullHamiltonian(model.mpo).matrix - offset * np.eye(2 ** model.n),
            FullHamiltonian(shifted_model.mpo).matrix,
            atol=1e-12
        )
