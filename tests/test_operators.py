import numpy as np
import pytest

from tnpy.model import XXZ, RandomHeisenberg
from tnpy.operators import FullHamiltonian, SpinOperators


class TestSpinOperators:
    def test_spin_half_ops(self):
        spin_half_ops = SpinOperators()
        np.testing.assert_array_equal(
            np.array([[0, 1], [1, 0]]), spin_half_ops.Sp + spin_half_ops.Sm
        )
        np.testing.assert_array_equal(
            np.array([[0, -1j], [1j, 0]]), -1j * (spin_half_ops.Sp - spin_half_ops.Sm)
        )


class TestMatrixProductOperator:
    @pytest.mark.parametrize("model", [RandomHeisenberg(n=4, h=0), RandomHeisenberg(n=4, h=0.5)])
    def test_square(self, model):
        bilayer_mpo = model.mpo.square()
        assert bilayer_mpo[0].shape == (25, 2, 2)
        assert bilayer_mpo[1].shape == (25, 25, 2, 2)
        assert bilayer_mpo[2].shape == (25, 25, 2, 2)
        assert bilayer_mpo[3].shape == (25, 2, 2)
        ham = FullHamiltonian(model.mpo).matrix
        np.testing.assert_allclose(ham @ ham, FullHamiltonian(bilayer_mpo).matrix, atol=1e-12)

    @pytest.mark.parametrize("n", [2, 4, 6])
    @pytest.mark.parametrize("h", [0, 0.5, 1])
    def test_multiply_scalar(self, n, h):
        mpo = RandomHeisenberg(n=n, h=h).mpo
        np.testing.assert_array_equal(
            -1 * FullHamiltonian(mpo).matrix, FullHamiltonian(-1 * mpo).matrix
        )


class TestFullHamiltonian:
    @pytest.fixture(
        scope="class",
        params=[
            {
                "n": 2,
                "ham": FullHamiltonian(RandomHeisenberg(n=2, h=0).mpo),
                "data": np.array(
                    [[0.25, 0, 0, 0], [0, -0.25, 0.5, 0], [0, 0.5, -0.25, 0], [0, 0, 0, 0.25]]
                ),
            },
            {
                "n": 3,
                "ham": FullHamiltonian(RandomHeisenberg(n=3, h=0).mpo),
                "data": np.array(
                    [
                        [0.5, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0.5, 0, 0, 0, 0, 0],
                        [0, 0.5, -0.5, 0, 0.5, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0.5, 0, 0],
                        [0, 0, 0.5, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0.5, 0, -0.5, 0.5, 0],
                        [0, 0, 0, 0, 0, 0.5, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0.5],
                    ]
                ),
            },
            {
                "n": 2,
                "ham": FullHamiltonian(XXZ(n=2, delta=0.5).mpo),
                "data": np.array(
                    [[-0.125, 0, 0, 0], [0, 0.125, -0.5, 0], [0, -0.5, 0.125, 0], [0, 0, 0, -0.125]]
                ),
            },
            {
                "n": 3,
                "ham": FullHamiltonian(XXZ(n=3, delta=0.5).mpo),
                "data": np.array(
                    [
                        [-0.25, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, -0.5, 0, 0, 0, 0, 0],
                        [0, -0.5, 0.25, 0, -0.5, 0, 0, 0],
                        [0, 0, 0, 0, 0, -0.5, 0, 0],
                        [0, 0, -0.5, 0, 0, 0, 0, 0],
                        [0, 0, 0, -0.5, 0, 0.25, -0.5, 0],
                        [0, 0, 0, 0, 0, -0.5, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, -0.25],
                    ]
                ),
            },
        ],
    )
    def model(self, request):
        return request.param

    def test_n_sites(self, model):
        assert model["ham"].n_sites == model["n"]

    def test_matrix(self, model):
        np.testing.assert_array_equal(model["ham"].matrix, model["data"])
