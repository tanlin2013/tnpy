import pytest
import numpy as np

from tnpy.exact_diagonalization import ExactDiagonalization
from tnpy.finite_dmrg import FiniteDMRG, ShiftInvertDMRG
from tnpy.model import XXZ, RandomHeisenberg


class TestFiniteDMRG:
    @pytest.fixture(scope="class")
    def model(self):
        return XXZ(n=10, delta=0.5)

    @pytest.fixture(scope="class")
    def ed(self, model):
        return ExactDiagonalization(model.mpo)

    @pytest.fixture(scope="function")
    def fdmrg(self, model):
        return FiniteDMRG(model.mpo, bond_dim=2**5)

    def test_run(self, ed, fdmrg):
        np.testing.assert_allclose(ed.evals[0], fdmrg.run(tol=1e-8)[-1], atol=1e-8)


class TestShiftInvertDMRG:
    @pytest.fixture(scope="class", params=[0.1, 0.2])
    def offset(self, request) -> float:
        return request.param

    @pytest.fixture(scope="class")
    def model(self):
        return RandomHeisenberg(n=8, h=0.5, seed=2021)

    @pytest.fixture(scope="class")
    def ed(self, model):
        return ExactDiagonalization(model.mpo)

    @pytest.fixture(scope="function")
    def sidmrg(self, model, offset):
        shifted_model = RandomHeisenberg(
            n=model.n, h=model.h, seed=model.seed, offset=offset
        )
        return ShiftInvertDMRG(shifted_model.mpo, bond_dim=2**6, offset=offset)

    @pytest.fixture(scope="class")
    def nearest_idx(self, ed, offset):
        return ed.evals[np.where(ed.evals < offset)].argmax()

    @pytest.fixture(scope="class")
    def nearest_eval(self, ed, nearest_idx):
        return ed.evals[nearest_idx]

    @pytest.fixture(scope="class")
    def nearest_evec(self, ed, nearest_idx):
        return ed.evecs[:, nearest_idx]

    def test_run(self, ed, sidmrg, nearest_eval):
        np.testing.assert_allclose(sidmrg.run(tol=1e-8)[-1], nearest_eval, atol=1e-6)

    def test_restored_mps(self, sidmrg, nearest_evec):
        sidmrg.run(tol=1e-8)
        restored_vec = (
            sidmrg.restored_mps.contract()
            .fuse({"k": sidmrg.restored_mps.outer_inds()})
            .data
        )
        # Note: They can differ up to a global phase (+1 or -1)
        if not np.allclose(restored_vec, nearest_evec, atol=1e-6):
            np.testing.assert_allclose(-1 * restored_vec, nearest_evec, atol=1e-6)

    def test_restore_energy(self, sidmrg, model):
        restored_energy = sidmrg.run(tol=1e-8)[-1]
        np.testing.assert_allclose(
            restored_energy, sidmrg.measurements.expectation_value(model.mpo), atol=1e-6
        )
