import imghdr
from itertools import product

import pytest
import numpy as np

from tnpy.model import RandomHeisenberg
from tnpy.exact_diagonalization import ExactDiagonalization
from tnpy.tsdrg import (
    TensorTree,
    TreeTensorNetworkSDRG as tSDRG,
    TreeTensorNetworkMeasurements
)


class TestTensorTree:

    @pytest.fixture(scope='class')
    def tree(self):
        model = RandomHeisenberg(n=4, h=0, penalty=0, s_target=0)
        tree = TensorTree(model.mpo)
        tree.fuse(
            left_id=1, right_id=2, new_id=4,
            data=ExactDiagonalization(
                RandomHeisenberg(n=2, h=0).mpo
            ).evecs.reshape((2, 2, 4))
        )
        tree.fuse(
            left_id=0, right_id=4, new_id=5,
            data=ExactDiagonalization(
                RandomHeisenberg(n=3, h=0).mpo
            ).evecs.reshape((2, 4, 8))
        )
        tree.fuse(
            left_id=5, right_id=3, new_id=6,
            data=ExactDiagonalization(
                RandomHeisenberg(n=4, h=0).mpo
            ).evecs.reshape((8, 2, 16))
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
            tree.fuse(
                left_id=6, right_id=5, new_id=7, data=np.random.rand(16, 16, 16)
            )

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
        net = tree.tensor_network(conj=True, mangle_outer=False)
        # net.graph()

    @pytest.mark.parametrize(
        "graph",
        ['digraph {\n\thead [style=invis]\n\thead -> 6 [arrowhead=inv headport=n]\n\t6 [rank=max shape=triangle '
        'style=rounded]\n\t5 [shape=triangle style=rounded]\n\t6 -> 5 [arrowhead=inv constraint=true headport=n '
        'minlen=2 splines=ortho tailport=_]\n\t5 -> 0 [arrowhead=inv constraint=true headport=n minlen=2 '
        'splines=ortho tailport=_]\n\t4 [shape=triangle style=rounded]\n\t5 -> 4 [arrowhead=inv constraint=true '
        'headport=n minlen=2 splines=ortho tailport=_]\n\t4 -> 1 [arrowhead=inv constraint=true headport=n '
        'minlen=2 splines=ortho tailport=_]\n\t4 -> 2 [arrowhead=inv constraint=true headport=n minlen=2 '
        'splines=ortho tailport=_]\n\t6 -> 3 [arrowhead=inv constraint=true headport=n minlen=2 splines=ortho '
        'tailport=_]\n\tsubgraph cluster_0 {\n\t\tstyle=invis\n\t\t0 [rank=sink shape=box style=rounded]\n\t\t0 '
        '-> 1 [arrowhead=none constraint=true minlen=0 splines=ortho]\n\t\t1 [rank=sink shape=box '
        'style=rounded]\n\t\t1 -> 2 [arrowhead=none constraint=true minlen=0 splines=ortho]\n\t\t2 [rank=sink '
        'shape=box style=rounded]\n\t\t2 -> 3 [arrowhead=none constraint=true minlen=0 splines=ortho]\n\t\t3 ['
        'rank=sink shape=box style=rounded]\n\t}\n}\n']
    )
    def test_plot(self, tree, graph):
        g = tree.plot()
        assert str(g) == graph
        # @Note: Method Digraph.render() doesn't work in command-line environment
        # g.render(format='png', view=True)


class TestTreeTensorNetworkSDRG:

    @pytest.fixture(scope='class')
    def model(self):
        return RandomHeisenberg(n=10, h=0.0, penalty=0, s_target=0)

    @pytest.fixture(scope='class')
    def tsdrg(self, model):
        return tSDRG(model.mpo, chi=2 ** 6)

    @pytest.fixture(scope='class')
    def ed(self, model):
        return ExactDiagonalization(model.mpo)

    @pytest.mark.parametrize("site", list(range(5)))
    def test_block_hamiltonian(self, tsdrg, site):
        np.testing.assert_array_equal(
            np.array(
                [[0.25, 0, 0, 0],
                 [0, -0.25, 0.5, 0],
                 [0, 0.5, -0.25, 0],
                 [0, 0, 0, 0.25]]
            ),
            tsdrg.block_hamiltonian(site)
        )

    def test_spectrum_projector(self, tsdrg):
        evecs = np.array(
            [[0, 1, 0, 0],
             [1 / np.sqrt(2), 0, 1 / np.sqrt(2), 0],
             [-1 / np.sqrt(2), 0, 1 / np.sqrt(2), 0],
             [0, 0, 0, 1]]
        )
        projector = tsdrg.spectrum_projector(locus=3, evecs=evecs)
        np.testing.assert_array_equal(
            evecs.reshape((2, 2, 4)),
            projector
        )

    def test_run(self):
        import sys
        np.set_printoptions(threshold=sys.maxsize)
        self.tsdrg.run()
        evals = self.tsdrg.evals
        idx = np.argsort(evals)
        # print(self.ed.evals[np.where(np.logical_and(self.ed.evals >= min(evals), self.ed.evals <= max(evals)))])
        print(evals[idx])
        var = self.tsdrg.measurements.expectation_value(self.tsdrg.mpo.square()) \
            - np.square(self.tsdrg.measurements.expectation_value(self.tsdrg.mpo))
        print(var[idx])
        self.tsdrg.tree.plot().render(format='png', view=True)
        # np.testing.assert_allclose(
        #     self.ed.evals[:self.tsdrg.chi],
        #     # self.tsdrg.measurements.expectation_value(self.tsdrg.mpo),
        #     evals[idx],
        #     atol=1e-12
        # )


class TestTreeTensorNetworkMeasurements:

    def __init__(self, *args, **kwargs):
        super(TestTreeTensorNetworkMeasurements, self).__init__(*args, **kwargs)
        self.model = RandomHeisenberg(n=8, h=0.5, penalty=0, s_target=0)
        self.chi = 2 ** 8
        self.tsdrg = tSDRG(self.model.mpo, self.chi)
        self.tsdrg.run()
        self.measurer = TreeTensorNetworkMeasurements(self.tsdrg.tree)
        self.ed = ExactDiagonalization(self.model.mpo)

    def test_sandwich(self):
        np.testing.assert_allclose(
            self.tsdrg.evals,
            np.diag(self.measurer.sandwich(self.tsdrg.mpo).data),
            atol=1e-12
        )
        np.testing.assert_allclose(
            np.identity(self.chi),
            self.measurer.sandwich().data,
            atol=1e-12
        )

    def test_expectation_value(self):
        spectral_folding_tsdrg = tSDRG(self.model.mpo.square(), self.chi)
        spectral_folding_tsdrg.run()
        np.testing.assert_allclose(
            self.ed.evals,
            np.sort(spectral_folding_tsdrg.measurements.expectation_value(self.model.mpo)),
            atol=1e-12
        )

    def test_min_surface(self):
        for site in range(self.tsdrg.n_sites - 1):
            min_side, surface, _ = self.measurer._min_surface(site)
            # print(site, surface)

    def test_reduced_density_matrix(self):
        # TODO: With the absence of disorder, h = 0, these tests fail for level_idx > 0
        # self.tsdrg.tree.plot().render(format='png', view=True)
        for site, level_idx in product(range(self.tsdrg.n_sites - 1), range(self.chi)[:1]):
            # print(site, level_idx)
            np.testing.assert_allclose(
                self.ed.reduced_density_matrix(site=site, level_idx=level_idx),
                np.linalg.eigvalsh(self.measurer.reduced_density_matrix(site=site, level_idx=level_idx))[::-1],
                atol=1e-12
            )

    def test_entanglement_entropy(self):
        for site, level_idx in product(range(self.tsdrg.n_sites - 1), range(self.chi)[:1]):
            # print(site, level_idx)
            self.assertAlmostEqual(
                self.ed.entanglement_entropy(site=site, level_idx=level_idx, nan_to_num=True),
                self.measurer.entanglement_entropy(site=site, level_idx=level_idx, nan_to_num=True),
                places=12
            )
