from unittest import TestCase
import numpy as np
import scipy.linalg as spla
from tnpy.model import RandomHeisenberg
from tnpy.exact_diagonalization import ExactDiagonalization
from tnpy.operators import FullHamiltonian


class TestExactDiagonalization(TestCase):

    def __init__(self, *args, **kwargs):
        super(TestExactDiagonalization, self).__init__(*args, **kwargs)
        self.ed = ExactDiagonalization(RandomHeisenberg(n=2, h=0).mpo)

    def test_matrix(self):
        np.testing.assert_array_equal(
            np.array(
                [[0.25, 0, 0, 0],
                 [0, -0.25, 0.5, 0],
                 [0, 0.5, -0.25, 0],
                 [0, 0, 0, 0.25]]
            ),
            self.ed.matrix
        )

    def test_evals(self):
        np.testing.assert_array_equal(
            np.array([-0.75, 0.25, 0.25, 0.25]),
            self.ed.evals
        )

    def test_evecs(self):
        np.testing.assert_array_equal(
            np.array(
                [[0, 1, 0, 0],
                 [1/np.sqrt(2), 0, 1/np.sqrt(2), 0],
                 [-1/np.sqrt(2), 0, 1/np.sqrt(2), 0],
                 [0, 0, 0, 1]]
            ),
            self.ed.evecs
        )


class TestShiftInvertExactDiagonalization(TestCase):

    def __init__(self, *args, **kwargs):
        super(TestShiftInvertExactDiagonalization, self).__init__(*args, **kwargs)
        model = RandomHeisenberg(n=8, h=0.5, penalty=0, s_target=0, seed=2021)
        self.ham = FullHamiltonian(model.mpo).matrix
        self.evals, self.evecs = np.linalg.eigh(self.ham)

        self.offset = 0.2
        shifted_model = RandomHeisenberg(
            n=model.n, h=model.h,
            penalty=model.penalty, s_target=model.s_target,
            seed=model.seed, offset=self.offset
        )
        self.shifted_ham = FullHamiltonian(shifted_model.mpo).matrix
        self.sf_evals, sf_evecs = spla.eigh(
            self.shifted_ham,
            b=self.shifted_ham @ self.shifted_ham
        )
        self.restored_evals = 1 / self.sf_evals + self.offset
        self.restored_evecs = self.shifted_ham @ sf_evecs

    def test_restored_evals(self):
        np.testing.assert_allclose(
            self.restored_evals,
            np.diag(self.restored_evecs.T @ self.ham @ self.restored_evecs),
            atol=1e-12
        )
        np.testing.assert_allclose(
            np.sort(self.restored_evals),
            self.evals,
            atol=1e-12
        )

    def test_restored_evecs(self):
        idx = np.argsort(self.restored_evals)
        np.testing.assert_allclose(
            self.restored_evecs[:, idx],
            self.evecs,
            atol=1e-12
        )

    def test_residual(self):
        res = np.array([
            self.ham @ self.restored_evecs[:, i] - self.restored_evals[i] * self.restored_evecs[:, i]
            for i in range(len(self.restored_evals))
        ])
        np.testing.assert_allclose(
            np.linalg.norm(res, axis=0),
            np.zeros(len(self.restored_evals)),
            atol=1e-12
        )

    # def test_visualize(self):
    #     import matplotlib.pyplot as plt
    #     plt.figure()
    #     plt.plot(self.restored_evals, self.sf_evals, marker='o')
    #     plt.show()
