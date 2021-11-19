import unittest
import numpy as np
from tensornetwork import Node, ncon
from tnpy.tsdrg import TreeNode, TensorTree, TSDRG
from tnpy.exact_diagonalization import ExactDiagonalization as ed
from tnpy.model import RandomHeisenberg


class TestTensorTree(unittest.TestCase):

    model = RandomHeisenberg(N=4, h=0, penalty=0.0, s_target=0)
    node0 = TreeNode(0, model.mpo[0])
    node1 = TreeNode(1, model.mpo[1])
    node2 = TreeNode(2, model.mpo[2])
    node3 = TreeNode(3, model.mpo[3])
    node4 = Node(ed(RandomHeisenberg(N=2, h=0).mpo).evecs.reshape((2, 2, 4)))
    node5 = Node(ed(RandomHeisenberg(N=3, h=0).mpo).evecs.reshape((2, 4, 8)))
    node6 = Node(ed(RandomHeisenberg(N=4, h=0).mpo).evecs.reshape((8, 2, 16)))

    def __init__(self, *args, **kwargs):
        super(TestTensorTree, self).__init__(*args, **kwargs)
        self.tree = TensorTree([self.node0, self.node1, self.node2, self.node3])
        # test TensorTree.append() and TensorTree.horizon
        self.tree.append(TreeNode(4, self.node4, left=self.node1, right=self.node2))
        self.assertCountEqual([0, 4, 3], self.tree.horizon)
        self.tree.append(TreeNode(5, self.node5, left=self.node0, right=self.tree.node(4)))
        self.assertCountEqual([5, 3], self.tree.horizon)
        self.tree.append(TreeNode(6, self.node6, left=self.tree.node(5), right=self.node3))
        self.assertCountEqual([6], self.tree.horizon)

    def test_is_leaf(self):
        self.assertTrue(self.node0.is_leaf)
        self.assertTrue(self.node1.is_leaf)
        self.assertTrue(self.node2.is_leaf)
        self.assertTrue(self.node3.is_leaf)
        self.assertFalse(self.tree.node(5).is_leaf)

    def test_equal(self):
        self.assertTrue(self.node0 == TreeNode(0, None))
        self.assertFalse(self.tree.node(6) == TreeNode(6, None))

    def test_n_nodes(self):
        self.assertEqual(7, self.tree.n_nodes)

    def test_n_layers(self):
        self.assertEqual(4, self.tree.n_layers)

    def test_has_root(self):
        self.assertTrue(self.tree.has_root)
        empty_tree = TensorTree([])
        self.assertFalse(empty_tree.has_root)

    def test_node(self):
        self.assertTrue(self.tree.node(5) == self.tree.root.left)
        self.assertTrue(self.tree.node(4) == self.tree.node(5).right)

    def test_find_path(self):
        self.assertCountEqual([6, 5, 0], self.tree.find_path(0))
        self.assertCountEqual([6, 5, 4], self.tree.find_path(4))
        self.assertCountEqual([6, 5, 4, 2], self.tree.find_path(2))
        self.assertRaises(KeyError, self.tree.find_path, 7)
        self.assertRaises(KeyError, self.tree.find_path, -1)

    def test_ancestor(self):
        ancestor = self.tree.ancestor(2)
        self.assertEqual(3, len(ancestor))
        self.assertTrue(ancestor[0] == self.tree.root)
        self.assertTrue(ancestor[1] == self.tree.node(5))
        self.assertTrue(ancestor[2] == self.tree.node(4))

    def test_common_ancestor(self):
        self.assertCountEqual([6, 5, 4], self.tree.common_ancestor(1, 2))
        self.assertCountEqual([6, 5], self.tree.common_ancestor(0, 2))
        self.assertCountEqual([6, 5], self.tree.common_ancestor(0, 4))
        self.assertCountEqual([6], self.tree.common_ancestor(2, 3))

    def test_contract_nodes(self):
        np.testing.assert_allclose(
            np.identity(16),
            self.tree.contract_nodes([4, 5, 6]),
            atol=1e-12
        )
        np.testing.assert_allclose(
            np.identity(16),
            self.tree.contract_nodes([5, 6]),
            atol=1e-12
        )
        np.testing.assert_allclose(
            np.identity(16),
            self.tree.contract_nodes([6]),
            atol=1e-12
        )
        out_tensor, out_order = self.tree.contract_nodes(
            [6, 5, 4],
            open_bonds=[(5, 'left')],
            return_out_order=True
        )
        self.assertCountEqual(
            (16, 2, 16, 2),
            out_tensor.shape
        )
        self.assertListEqual(
            ["-Node6Sub2", "-Node5Sub0", "-ConjNode6Sub2", "-ConjNode5Sub0"],
            out_order
        )
        out_tensor, out_order = self.tree.contract_nodes(
            [6, 5],
            open_bonds=[(5, 'right')],
            return_out_order=True
        )
        self.assertCountEqual(
            (16, 4, 16, 4),
            out_tensor.shape
        )
        self.assertListEqual(
            ["-Node6Sub2", "-Node5Sub1", "-ConjNode6Sub2", "-ConjNode5Sub1"],
            out_order
        )


class TestTSDRG(unittest.TestCase):

    model = RandomHeisenberg(N=6, h=0, penalty=0.0, s_target=0)
    tsdrg = TSDRG(model.mpo, chi=2**6)

    def test_N(self):
        self.assertEqual(6, self.tsdrg.N)

    def test_block_hamiltonian(self):
        for site in range(self.tsdrg.N - 1):
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
        V, W = self.tsdrg.spectrum_projector(site=3, evecs=evecs)
        np.testing.assert_array_equal(
            evecs.reshape((2, 2, 4)),
            V.tensor
        )
        coarse_grained_mpo = ncon(
            [self.model.mpo[3].tensor, self.model.mpo[4].tensor],
            [(-1, 1, '-a1', '-a2'), (1, -2, '-b1', '-b2')],
            out_order=[-1, -2, '-a1', '-b1', '-a2', '-b2']
        ).reshape((6, 6, 4, 4))
        np.testing.assert_allclose(
            ncon(
                [coarse_grained_mpo, evecs, evecs],
                [('-m1', '-m2', 1, 2), (1, '-a1'), (2, '-b1')],
                out_order=['-m1', '-m2', '-a1', '-b1']
            ),
            W.tensor,
            atol=1e-12
        )


if __name__ == '__main__':
    unittest.main()
