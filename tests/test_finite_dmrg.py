from pprint import pprint

import pytest
import numpy as np
from tnpy.exact_diagonalization import ExactDiagonalization
from tnpy.finite_dmrg import FiniteDMRG, ShiftInvertDMRG, Metric
from tnpy.model import XXZ, RandomHeisenberg


class TestFiniteDMRG:

    @pytest.fixture(scope='class')
    def model(self):
        return XXZ(n=10, delta=0.5)

    @pytest.fixture(scope='class')
    def ed(self, model):
        return ExactDiagonalization(model.mpo)

    @pytest.fixture(scope='class')
    def fdmrg(self, model):
        return FiniteDMRG(model.mpo, bond_dim=2**5)

    def test_run(self, ed, fdmrg):
        assert np.allclose(
            ed.evals[0],
            fdmrg.run(tol=1e-8)[-1],
            atol=1e-8
        )


class TestShiftInvertDMRG:

    def __init__(self, *args, **kwargs):
        super(TestShiftInvertDMRG, self).__init__(*args, **kwargs)
        self.offset = 0.1
        self.model = RandomHeisenberg(n=8, h=0.5, seed=2021)
        self.shifted_model = RandomHeisenberg(
            n=self.model.n, h=self.model.h,
            seed=self.model.seed, offset=self.offset
        )
        self.ed = ExactDiagonalization(self.model.mpo)
        self.sidmrg = ShiftInvertDMRG(self.shifted_model.mpo, bond_dim=2 ** 3, offset=self.offset)
        self.nearest_idx = self.ed.evals[np.where(self.ed.evals < self.offset)].argmax()

    @property
    def nearest_eval(self):
        return self.ed.evals[self.nearest_idx]

    @property
    def nearest_evec(self):
        return self.ed.evecs[:, self.nearest_idx]

    def test_run(self):
        print(self.ed.evals)
        np.testing.assert_almost_equal(
            self.sidmrg.run(tol=1e-8)[-1],
            self.nearest_eval,
            decimal=8
        )

    def test_restored_mps(self):
        self.sidmrg.run(tol=1e-8)
        restored_vec = self.sidmrg.restored_mps.contract().fuse(
            {'k': self.sidmrg.restored_mps.outer_inds()}
        ).data
        res = restored_vec - self.nearest_evec
        res[np.abs(res) < 1e-8] = 0
        print(res)
        if not np.allclose(restored_vec, self.nearest_evec, atol=1e-8):
            np.testing.assert_allclose(
                -1 * restored_vec,
                self.nearest_evec,
                atol=1e-8
            )

    def test_restore_energy(self):
        restored_energy = self.sidmrg.run(tol=1e-8)[-1]
        np.testing.assert_almost_equal(
            restored_energy,
            self.sidmrg.measurements.expectation_value(self.model.mpo),
            decimal=8
        )
