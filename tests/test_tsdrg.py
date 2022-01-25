from unittest import TestCase
import numpy as np
from itertools import product
from tnpy.model import RandomHeisenberg
from tnpy.exact_diagonalization import ExactDiagonalization
from tnpy.tsdrg import (
    TensorTree,
    TreeTensorNetworkSDRG as tSDRG,
    TreeTensorNetworkMeasurements
)


class TestTensorTree(TestCase):

    def __init__(self, *args, **kwargs):
        super(TestTensorTree, self).__init__(*args, **kwargs)
        model = RandomHeisenberg(n=4, h=0, penalty=0, s_target=0)
        self.tree = TensorTree(model.mpo)
        self.tree.fuse(
            left_id=1, right_id=2, new_id=4,
            data=ExactDiagonalization(RandomHeisenberg(n=2, h=0).mpo).evecs.reshape((2, 2, 4))
        )
        self.tree.fuse(
            left_id=0, right_id=4, new_id=5,
            data=ExactDiagonalization(RandomHeisenberg(n=3, h=0).mpo).evecs.reshape((2, 4, 8))
        )
        self.tree.fuse(
            left_id=5, right_id=3, new_id=6,
            data=ExactDiagonalization(RandomHeisenberg(n=4, h=0).mpo).evecs.reshape((8, 2, 16))
        )

    def test_is_leaf(self):
        self.assertTrue(self.tree[0].is_leaf)
        self.assertTrue(self.tree[1].is_leaf)
        self.assertTrue(self.tree[2].is_leaf)
        self.assertTrue(self.tree[3].is_leaf)
        self.assertFalse(self.tree[5].is_leaf)

    def test_n_nodes(self):
        self.assertEqual(7, self.tree.n_nodes)

    def test_n_layers(self):
        self.assertEqual(4, self.tree.n_layers)

    def test_getitem(self):
        self.assertTrue(self.tree[5] == self.tree.root.left)
        self.assertTrue(self.tree[4] == self.tree[5].right)

    def test_fuse(self):
        self.assertRaises(
            RuntimeError,
            self.tree.fuse,
            left_id=6, right_id=5, new_id=7, data=np.random.rand(16, 16, 16)
        )

    def test_find_path(self):
        self.assertCountEqual([6, 5], self.tree.find_path(0))
        self.assertCountEqual([6, 5], self.tree.find_path(4))
        self.assertCountEqual([6, 5, 4], self.tree.find_path(2))
        self.assertRaises(KeyError, self.tree.find_path, 7)
        self.assertRaises(KeyError, self.tree.find_path, -1)

    def test_common_ancestor(self):
        self.assertCountEqual([6, 5, 4], self.tree.common_ancestor(1, 2))
        self.assertCountEqual([6, 5], self.tree.common_ancestor(0, 2))
        self.assertCountEqual([6, 5], self.tree.common_ancestor(0, 4))
        self.assertCountEqual([6], self.tree.common_ancestor(2, 3))
        self.assertEqual(5, self.tree.common_ancestor(0, 1, lowest=True))
        self.assertEqual(6, self.tree.common_ancestor(1, 3, lowest=True))

    def test_tensor_network(self):
        net = self.tree.tensor_network(conj=True, mangle_outer=False)
        # net.graph()

    def test_plot(self):
        g = self.tree.plot()
        self.assertEqual(
            str(g),
            'digraph {\n\thead [style=invis]\n\thead -> 6 [arrowhead=inv headport=n]\n\t6 [rank=max shape=triangle '
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
            'rank=sink shape=box style=rounded]\n\t}\n}\n'
        )
        # @Note: Method Digraph.render() doesn't work in command-line environment
        # g.render(format='png', view=True)


class TestTreeTensorNetworkSDRG(TestCase):

    def __init__(self, *args, **kwargs):
        super(TestTreeTensorNetworkSDRG, self).__init__(*args, **kwargs)
        model = RandomHeisenberg(n=6, h=0.0, penalty=0, s_target=0)
        self.chi = 2 ** 6
        self.tsdrg = tSDRG(model.mpo, self.chi)
        self.ed = ExactDiagonalization(model.mpo)

    def test_block_hamiltonian(self):
        for site in range(self.tsdrg.n_sites - 1):
            np.testing.assert_array_equal(
                np.array(
                    [[0.25, 0, 0, 0],
                     [0, -0.25, 0.5, 0],
                     [0, 0.5, -0.25, 0],
                     [0, 0, 0, 0.25]]
                ),
                self.tsdrg.block_hamiltonian(site)
            )

    def test_spectrum_projector(self):
        evecs = np.array(
            [[0, 1, 0, 0],
             [1 / np.sqrt(2), 0, 1 / np.sqrt(2), 0],
             [-1 / np.sqrt(2), 0, 1 / np.sqrt(2), 0],
             [0, 0, 0, 1]]
        )
        projector = self.tsdrg.spectrum_projector(locus=3, evecs=evecs)
        np.testing.assert_array_equal(
            evecs.reshape((2, 2, 4)),
            projector
        )

    def test_run(self):
        self.tsdrg.run()
        np.testing.assert_allclose(
            self.ed.evals[:self.tsdrg.chi],
            self.tsdrg.evals,
            atol=1e-12
        )


class TestTreeTensorNetworkMeasurements(TestCase):

    def __init__(self, *args, **kwargs):
        super(TestTreeTensorNetworkMeasurements, self).__init__(*args, **kwargs)
        self.model = RandomHeisenberg(n=8, h=0.0, penalty=0, s_target=0)
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
            surface, _ = self.measurer._min_surface(site)

    def test_reduced_density_matrix(self):
        # TODO: With the absence of disorder, h = 0, these tests fail for level_idx > 0
        for site, level_idx in product(range(self.tsdrg.n_sites - 1), range(self.chi)[:1]):
            # print(site, level_idx)
            np.testing.assert_allclose(
                self.ed.reduced_density_matrix(site=site, level_idx=level_idx),
                np.linalg.eigvalsh(self.measurer.reduced_density_matrix(site=site, level_idx=level_idx))[::-1],
                atol=1e-12
            )

    def test_entanglement_entropy(self):
        for site, level_idx in product(range(self.tsdrg.n_sites - 1), range(self.chi)[:1]):
            self.assertAlmostEqual(
                self.ed.entanglement_entropy(site=site, level_idx=level_idx),
                self.measurer.entanglement_entropy(site=site, level_idx=level_idx),
                places=12
            )
