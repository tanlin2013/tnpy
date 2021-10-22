import unittest
from tnpy.tsdrg import TreeNode, TensorTree, TSDRG
from tnpy.model import RandomHeisenberg


class TestTensorTree(unittest.TestCase):

    node0 = TreeNode(0, None)
    node1 = TreeNode(1, None)
    node2 = TreeNode(2, None)
    node3 = TreeNode(3, None)

    def __init__(self, *args, **kwargs):
        super(TestTensorTree, self).__init__(*args, **kwargs)
        self.tree = TensorTree([self.node0, self.node1, self.node2, self.node3])
        # test TensorTree.append() and TensorTree.horizon
        self.tree.append(TreeNode(4, None, left=self.node1, right=self.node2))
        self.assertCountEqual([0, 4, 3], self.tree.horizon)
        self.tree.append(TreeNode(5, None, left=self.node0, right=self.tree.node(4)))
        self.assertCountEqual([5, 3], self.tree.horizon)
        self.tree.append(TreeNode(6, None, left=self.tree.node(5), right=self.node3))
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


if __name__ == '__main__':
    unittest.main()
