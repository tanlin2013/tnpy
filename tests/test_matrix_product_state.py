import os
from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest

from tnpy.matrix_product_state import Direction, Environment, MatrixProductState
from tnpy.model import RandomHeisenberg


class Helper:
    @staticmethod
    def compressed_bond_dims(n: int, bond_dim: int, phys_dim: int) -> np.ndarray:
        if n % 2 == 0:
            chi = [min(phys_dim**i, bond_dim) for i in range(1, n // 2)]
            chi += [int(min(phys_dim ** (n / 2), bond_dim))] + chi[::-1]
        else:
            chi = [min(phys_dim**i, bond_dim) for i in range(1, (n + 1) // 2)]
            chi += chi[::-1]
        return np.array(chi)


class TestMatrixProductState:
    @pytest.mark.parametrize("n", [6, 8])
    @pytest.mark.parametrize("bond_dim", [2, 4, 6])
    @pytest.mark.parametrize("phys_dim", [2, 4])
    def test_shape(self, n, bond_dim, phys_dim):
        mps = MatrixProductState.random(n=n, bond_dim=bond_dim, phys_dim=phys_dim)
        chi = Helper.compressed_bond_dims(n=n, bond_dim=bond_dim, phys_dim=phys_dim)
        assert (chi <= phys_dim ** (n // 2)).all()
        for site, tensor in enumerate(mps):
            if site == 0:
                assert tensor.shape == (phys_dim, chi[0])
            elif site == n - 1:
                assert tensor.shape == (chi[-1], phys_dim)
            else:
                assert tensor.shape == (chi[site - 1], phys_dim, chi[site])

    @pytest.fixture(scope="class")
    def mps(self) -> MatrixProductState:
        return MatrixProductState.random(n=8, bond_dim=10, phys_dim=2)

    def test_get_item(self, mps):
        mps.left_canonize()
        assert mps[0].shape == (2, 2)
        assert mps[1].shape == (2, 2, 4)
        assert mps[2].shape == (4, 2, 8)
        assert mps[-2].shape == (4, 2, 2)
        assert mps[-1].shape == (2, 2)

    def test_conj(self, mps):
        conj_mps = mps.conj(mangle_inner=True, mangle_outer=False)
        np.testing.assert_allclose(mps @ conj_mps, 1, atol=1e-12)
        conj_mps = mps.conj(mangle_inner=True, mangle_outer=True)
        assert len((mps @ conj_mps).inds) == 2 * mps.n_sites

    @pytest.mark.parametrize(
        "filename, expectation",
        [
            ("test.npz", does_not_raise()),
            ("test.hdf5", does_not_raise()),
            ("test.txt", pytest.raises(ValueError)),
        ],
    )
    def test_save(self, filename, expectation, mps):
        with expectation:
            mps.save(filename)
            assert os.path.isfile(filename)

    @pytest.mark.parametrize(
        "filename, expectation",
        [
            ("test.npz", does_not_raise()),
            ("test.hdf5", does_not_raise()),
            ("test.txt", pytest.raises(ValueError)),
        ],
    )
    def test_load(self, filename, expectation):
        with expectation:
            mps = MatrixProductState.load(filename)  # noqa: F841
            pass  # TODO: NotImplemented

    @pytest.mark.parametrize("site", [2, 3, 4, 6])
    def test_split_tensor(self, site, mps):
        two_site_mps = mps[site] @ mps[site + 1]
        mps.split_tensor(site, direction=Direction.RIGHTWARD)
        np.testing.assert_allclose(two_site_mps.data, (mps[site] @ mps[site + 1]).data, atol=1e-12)
        assert mps[site].tags == {f"I{site}"}
        assert mps[site + 1].tags == {f"I{site + 1}"}


class TestEnvironment:
    @pytest.fixture(scope="class")
    def env(self):
        model = RandomHeisenberg(n=6, h=0, penalty=100.0)
        return Environment(model.mpo, MatrixProductState.random(n=model.n, bond_dim=16, phys_dim=2))

    def test_left(self, env):
        print(env.mps)
        print(env.mpo)
        print(env._conj_mps)
        print(env.left)
        print(env.right)

    def test_right(self):
        pass

    def test_one_site_full_matrix(self):
        pass

    def test_one_site_matvec(self):
        pass
