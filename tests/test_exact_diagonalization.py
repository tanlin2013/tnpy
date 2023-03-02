import numpy as np
import pytest
import scipy.linalg as spla

from tnpy.exact_diagonalization import ExactDiagonalization
from tnpy.model import RandomHeisenberg
from tnpy.operators import FullHamiltonian


class TestExactDiagonalization:
    @pytest.fixture(scope="class")
    def ed(self):
        return ExactDiagonalization(RandomHeisenberg(n=2, h=0).mpo)

    def test_matrix(self, ed):
        np.testing.assert_array_equal(
            np.array(
                [
                    [0.25, 0, 0, 0],
                    [0, -0.25, 0.5, 0],
                    [0, 0.5, -0.25, 0],
                    [0, 0, 0, 0.25],
                ]
            ),
            ed.matrix,
        )

    def test_evals(self, ed):
        np.testing.assert_array_equal(np.array([-0.75, 0.25, 0.25, 0.25]), ed.evals)

    def test_evecs(self, ed):
        np.testing.assert_array_equal(
            np.array(
                [
                    [0, 1, 0, 0],
                    [1 / np.sqrt(2), 0, 1 / np.sqrt(2), 0],
                    [-1 / np.sqrt(2), 0, 1 / np.sqrt(2), 0],
                    [0, 0, 0, 1],
                ]
            ),
            ed.evecs,
        )


class TestShiftInvertExactDiagonalization:
    @pytest.fixture(scope="class", params=[0.1, 0.2, 0.4])
    def offset(self, request):
        return request.param

    @pytest.fixture(scope="class")
    def model(self):
        return RandomHeisenberg(n=8, h=0.5, penalty=0, s_target=0, seed=2021)

    @pytest.fixture(scope="class")
    def ham(self, model):
        return FullHamiltonian(model.mpo).matrix

    @pytest.fixture(scope="class")
    def eigen_solver(self, ham):
        return np.linalg.eigh(ham)

    @pytest.fixture(scope="class")
    def shift_inverted_model(self, model, offset):
        return RandomHeisenberg(
            n=model.n,
            h=model.h,
            penalty=model.penalty,
            s_target=model.s_target,
            seed=model.seed,
            offset=offset,
        )

    @pytest.fixture(scope="class")
    def shift_inverted_ham(self, shift_inverted_model):
        return FullHamiltonian(shift_inverted_model.mpo).matrix

    @pytest.fixture(scope="class")
    def shift_invert_eigen_solver(self, shift_inverted_ham, offset):
        sf_inv_evals, sf_inv_evecs = spla.eigh(
            shift_inverted_ham, b=shift_inverted_ham @ shift_inverted_ham
        )
        restored_evals = 1 / sf_inv_evals + offset
        restored_evecs = shift_inverted_ham @ sf_inv_evecs
        return restored_evals, restored_evecs

    def test_restored_evals(self, ham, eigen_solver, shift_invert_eigen_solver):
        evals, _ = eigen_solver
        restored_evals, restored_evecs = shift_invert_eigen_solver
        np.testing.assert_allclose(
            restored_evals,
            np.diag(restored_evecs.T @ ham @ restored_evecs),
            atol=1e-12,
        )
        np.testing.assert_allclose(np.sort(restored_evals), evals, atol=1e-12)

    def test_restored_evecs(self, eigen_solver, shift_invert_eigen_solver):
        _, evecs = eigen_solver
        restored_evals, restored_evecs = shift_invert_eigen_solver
        idx = np.argsort(restored_evals)
        # Note: This will fail, because they can differ up to a global phase (+1 or -1)
        assert not np.allclose(restored_evecs[:, idx], evecs, atol=1e-12)
        fidelity = np.square(np.diag(restored_evecs[:, idx].T @ evecs))
        np.testing.assert_allclose(fidelity, np.ones(2**8), atol=1e-12)

    def test_residual(self, ham, shift_invert_eigen_solver):
        restored_evals, restored_evecs = shift_invert_eigen_solver
        res = np.array(
            [
                ham @ restored_evecs[:, i] - restored_evals[i] * restored_evecs[:, i]
                for i in range(len(restored_evals))
            ]
        )
        np.testing.assert_allclose(
            np.linalg.norm(res, axis=0), np.zeros(len(restored_evals)), atol=1e-12
        )

    # def test_visualize(self):
    #     import matplotlib.pyplot as plt
    #     plt.figure()
    #     plt.plot(self.restored_evals, self.sf_evals, marker='o')
    #     plt.show()
