import pytest
import numpy as np

from tnpy.model import RandomHeisenberg, TotalSz
from tnpy.operators import SpinOperators, FullHamiltonian
from tnpy.exact_diagonalization import ExactDiagonalization
from tnpy.tsdrg import (
    TensorTree,
    TreeTensorNetworkSDRG as tSDRG,
    HighEnergyTreeTensorNetworkSDRG,
)


class TestTensorTree:
    @pytest.fixture(scope="class")
    def tree(self):
        model = RandomHeisenberg(n=4, h=0, penalty=0, s_target=0)
        tree = TensorTree(model.mpo)
        tree.fuse(
            left_id=1,
            right_id=2,
            new_id=4,
            data=ExactDiagonalization(RandomHeisenberg(n=2, h=0).mpo).evecs.reshape(
                (2, 2, 4)
            ),
        )
        tree.fuse(
            left_id=0,
            right_id=4,
            new_id=5,
            data=ExactDiagonalization(RandomHeisenberg(n=3, h=0).mpo).evecs.reshape(
                (2, 4, 8)
            ),
        )
        tree.fuse(
            left_id=5,
            right_id=3,
            new_id=6,
            data=ExactDiagonalization(RandomHeisenberg(n=4, h=0).mpo).evecs.reshape(
                (8, 2, 16)
            ),
        )
        return tree

    def test_is_leaf(self, tree):
        assert tree[0].is_leaf
        assert tree[1].is_leaf
        assert tree[2].is_leaf
        assert tree[3].is_leaf
        assert not tree[5].is_leaf

    def test_n_nodes(self, tree):
        assert tree.n_nodes == 7

    def test_n_layers(self, tree):
        assert tree.n_layers == 4

    def test_getitem(self, tree):
        assert tree.root.left == tree[5]
        assert tree[5].right == tree[4]

    def test_fuse(self, tree):
        with pytest.raises(RuntimeError):
            tree.fuse(left_id=6, right_id=5, new_id=7, data=np.random.rand(16, 16, 16))

    def test_find_path(self, tree):
        assert tree.find_path(0) == [6, 5]
        assert tree.find_path(4) == [6, 5]
        assert tree.find_path(2) == [6, 5, 4]
        with pytest.raises(KeyError):
            tree.find_path(7)
        with pytest.raises(KeyError):
            tree.find_path(-1)

    def test_common_ancestor(self, tree):
        assert tree.common_ancestor(1, 2) == [6, 5, 4]
        assert tree.common_ancestor(0, 2) == [6, 5]
        assert tree.common_ancestor(0, 4) == [6, 5]
        assert tree.common_ancestor(2, 3) == [6]
        assert tree.common_ancestor(0, 1, lowest=True) == 5
        assert tree.common_ancestor(1, 3, lowest=True) == 6

    def test_tensor_network(self, tree):
        net = tree.tensor_network(conj=True, mangle_outer=False)  # noqa: F841
        # net.graph()

    def test_plot(self, tree):
        graph = tree.plot()
        assert "head -> 6" in str(graph)
        assert "5 -> 0" in str(graph)
        assert "5 -> 4" in str(graph)
        assert "4 -> 1" in str(graph)
        assert "4 -> 2" in str(graph)
        assert "6 -> 3" in str(graph)
        assert "subgraph cluster_0" in str(graph)
        assert "0 -> 1" in str(graph)
        assert "1 -> 2" in str(graph)
        assert "2 -> 3" in str(graph)
        # @Note: Method Digraph.render() doesn't work in command-line environment
        # graph.render(format="png", view=True)


class TestTreeTensorNetworkSDRG:
    @pytest.fixture(scope="class")
    def model(self):
        return RandomHeisenberg(n=8, h=10.0, penalty=0, s_target=0)

    @pytest.fixture(scope="class")
    def tsdrg(self, model):
        return tSDRG(model.mpo, chi=2**6)

    @pytest.fixture(scope="class")
    def ed(self, model):
        return ExactDiagonalization(model.mpo)

    @pytest.fixture(scope="class")
    def static_tsdrg(self):
        model = RandomHeisenberg(n=8, h=0.0, penalty=0, s_target=0)
        return tSDRG(model.mpo, chi=2**6)

    @pytest.mark.parametrize("site", list(range(5)))
    def test_block_hamiltonian(self, static_tsdrg, site):
        np.testing.assert_array_equal(
            np.array(
                [
                    [0.25, 0, 0, 0],
                    [0, -0.25, 0.5, 0],
                    [0, 0.5, -0.25, 0],
                    [0, 0, 0, 0.25],
                ]
            ),
            static_tsdrg.block_hamiltonian(site),
        )

    def test_spectrum_projector(self, static_tsdrg):
        evecs = np.array(
            [
                [0, 1, 0, 0],
                [1 / np.sqrt(2), 0, 1 / np.sqrt(2), 0],
                [-1 / np.sqrt(2), 0, 1 / np.sqrt(2), 0],
                [0, 0, 0, 1],
            ]
        )
        projector = static_tsdrg.spectrum_projector(locus=3, evecs=evecs)
        np.testing.assert_array_equal(evecs.reshape((2, 2, 4)), projector)

    def test_run(self, ed, tsdrg):
        tsdrg.run()
        np.testing.assert_allclose(
            ed.evals[: tsdrg.chi], np.sort(tsdrg.evals), atol=1e-8
        )

    def test_measurements(self, ed, tsdrg):
        tsdrg.run()
        np.testing.assert_allclose(
            ed.evals[: tsdrg.chi],
            tsdrg.measurements.expectation_value(tsdrg.mpo),
            atol=1e-8,
        )

    def test_upside_down_spectrum(self, model, ed):
        tsdrg = tSDRG(-1 * model.mpo, chi=2**4)
        tsdrg.run()
        np.testing.assert_allclose(ed.evals[-1], -1 * tsdrg.evals[0], atol=1e-4)
        np.testing.assert_allclose(
            ed.evals[-1], tsdrg.measurements.expectation_value(model.mpo)[0], atol=1e-4
        )


class TestTreeTensorNetworkMeasurements:
    @pytest.fixture(scope="class")
    def model(self):
        return RandomHeisenberg(n=8, h=0.5, penalty=0, s_target=0)

    @pytest.fixture(scope="class")
    def tsdrg(self, model):
        tsdrg = tSDRG(model.mpo, chi=2**8)
        tsdrg.run()
        return tsdrg

    @pytest.fixture(scope="class")
    def ed(self, model):
        return ExactDiagonalization(model.mpo)

    def test_sandwich(self, tsdrg):
        np.testing.assert_allclose(
            tsdrg.evals,
            np.diag(tsdrg.measurements.sandwich(tsdrg.mpo).data),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            np.identity(tsdrg.chi), tsdrg.measurements.sandwich().data, atol=1e-12
        )

    def test_expectation_value(self, model, ed):
        spectral_folding_tsdrg = tSDRG(model.mpo.square(), chi=2**8)
        spectral_folding_tsdrg.run()
        np.testing.assert_allclose(
            ed.evals,
            np.sort(spectral_folding_tsdrg.measurements.expectation_value(model.mpo)),
            atol=1e-12,
        )

    @pytest.mark.parametrize("site", list(range(8 - 1)))
    def test_min_surface(self, site, tsdrg):
        min_side, surface, _ = tsdrg.measurements._min_surface(site)
        # print(site, surface)

    @pytest.mark.parametrize("site", list(range(8 - 1)))
    @pytest.mark.parametrize("level_idx", list(range(2**8))[:1])
    def test_reduced_density_matrix(self, site, level_idx, ed, tsdrg):
        # TODO: With the absence of disorder, h = 0, these tests fail for level_idx > 0
        # self.tsdrg.tree.plot().render(format='png', view=True)
        np.testing.assert_allclose(
            ed.reduced_density_matrix(site=site, level_idx=level_idx),
            np.linalg.eigvalsh(
                tsdrg.measurements.reduced_density_matrix(
                    site=site, level_idx=level_idx
                )
            )[::-1],
            atol=1e-12,
        )

    @pytest.mark.parametrize("site", list(range(8 - 1)))
    @pytest.mark.parametrize("level_idx", list(range(2**8))[:1])
    def test_entanglement_entropy(self, site, level_idx, ed, tsdrg):
        np.testing.assert_allclose(
            ed.entanglement_entropy(site=site, level_idx=level_idx, nan_to_num=True),
            tsdrg.measurements.entanglement_entropy(
                site=site, level_idx=level_idx, nan_to_num=True
            ),
            atol=1e-12,
        )

    @pytest.mark.parametrize("site", list(range(8 - 1)))
    @pytest.mark.parametrize("level_idx", [0])
    def test_one_point_function(self, site, level_idx, ed, tsdrg):
        Sp, Sm, Sz, I2, O2 = SpinOperators()
        np.testing.assert_allclose(
            ed.one_point_function(Sz, site, level_idx=level_idx),
            tsdrg.measurements.one_point_function(Sz, site, level_idx=level_idx),
            atol=1e-12,
        )

    @pytest.mark.parametrize("site1, site2", [(1, 3), (2, 6), (4, 3)])
    @pytest.mark.parametrize("level_idx", [0])
    def test_two_point_function(self, site1, site2, level_idx, ed, tsdrg):
        Sp, Sm, Sz, I2, O2 = SpinOperators()
        np.testing.assert_allclose(
            ed.two_point_function(Sz, Sz, site1, site2, level_idx=level_idx),
            tsdrg.measurements.two_point_function(
                Sz, Sz, site1, site2, level_idx=level_idx
            ),
            atol=1e-12,
        )

    @pytest.mark.parametrize("site1, site2", [(1, 3), (2, 6)])
    @pytest.mark.parametrize("level_idx", [0])
    def test_connected_two_point_function(self, site1, site2, level_idx, ed, tsdrg):
        Sp, Sm, Sz, I2, O2 = SpinOperators()
        np.testing.assert_allclose(
            ed.connected_two_point_function(Sz, Sz, site1, site2, level_idx=level_idx),
            tsdrg.measurements.connected_two_point_function(
                Sz, Sz, site1, site2, level_idx=level_idx
            ),
            atol=1e-12,
        )

    @pytest.mark.parametrize("partition_site", list(range(8)))
    def test_variance(self, ed, tsdrg, partition_site):
        np.testing.assert_allclose(
            ed.variance(),
            tsdrg.measurements.variance(tsdrg.mpo),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            tsdrg.measurements.variance(tsdrg.mpo),
            np.zeros(2**8),
            atol=1e-12,
        )
        sz_a = TotalSz(n=8).subsystem_mpo(partition_site)
        np.testing.assert_allclose(
            ed.variance(FullHamiltonian(sz_a).matrix),
            tsdrg.measurements.variance(sz_a),
            atol=1e-12,
        )


class TestHighEnergyTreeTensorNetworkSDRG:
    @pytest.fixture(scope="class")
    def model(self):
        return RandomHeisenberg(n=10, h=0.5, penalty=0, s_target=0)

    @pytest.fixture(scope="class")
    def tsdrg(self, model):
        return HighEnergyTreeTensorNetworkSDRG(model.mpo, chi=2**4)

    @pytest.fixture(scope="class")
    def ed(self, model):
        return ExactDiagonalization(model.mpo)

    def test_run(self, ed, tsdrg):
        print(ed.evals)
        tsdrg.run()
        np.testing.assert_allclose(ed.evals[-1], tsdrg.evals[-1], atol=1e-6)
