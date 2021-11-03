import unittest
import numpy as np
from tnpy.operators import FullHamiltonian, SpinOperators
from tnpy.model import XXZ, RandomHeisenberg


class TestMPO(unittest.TestCase):

    model_set = [
        RandomHeisenberg(N=4, h=0).mpo,
        XXZ(N=6, delta=0.5).mpo
    ]

    def test_nodes(self):
        self.assertEqual(4, len(self.model_set[0].nodes))
        self.assertEqual(6, len(self.model_set[1].nodes))
        self.assertEqual((6, 2, 2), self.model_set[0].nodes[0].shape)
        self.assertEqual((6, 6, 2, 2), self.model_set[0].nodes[1].shape)
        self.assertEqual((5, 2, 2), self.model_set[1].nodes[0].shape)
        self.assertEqual((5, 5, 2, 2), self.model_set[1].nodes[1].shape)

    def test_physical_dimensions(self):
        self.assertEqual(2, self.model_set[0].physical_dimensions)
        self.assertEqual(2, self.model_set[1].physical_dimensions)

    def test_bond_dimensions(self):
        self.assertEqual(6, self.model_set[0].bond_dimensions)
        self.assertEqual(5, self.model_set[1].bond_dimensions)


class TestFullHamiltonian(unittest.TestCase):

    ham_set = [
        FullHamiltonian(RandomHeisenberg(N=2, h=0).mpo),
        FullHamiltonian(RandomHeisenberg(N=3, h=0).mpo),
        FullHamiltonian(XXZ(N=2, delta=0.5).mpo),
        FullHamiltonian(XXZ(N=3, delta=0.5).mpo)
    ]

    def test_N(self):
        self.assertEqual(2, self.ham_set[0].N)
        self.assertEqual(3, self.ham_set[1].N)
        self.assertEqual(2, self.ham_set[2].N)
        self.assertEqual(3, self.ham_set[3].N)

    def test_matrix(self):
        np.testing.assert_array_equal(
            np.array(
                [[0.25, 0, 0, 0],
                 [0, -0.25, 0.5, 0],
                 [0, 0.5, -0.25, 0],
                 [0, 0, 0, 0.25]]
            ),
            self.ham_set[0].matrix
        )
        np.testing.assert_array_equal(
            np.array(
                [[0.5, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0.5, 0, 0, 0, 0, 0],
                 [0, 0.5, -0.5, 0, 0.5, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0.5, 0, 0],
                 [0, 0, 0.5, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0.5, 0, -0.5, 0.5, 0],
                 [0, 0, 0, 0, 0, 0.5, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0.5]]
            ),
            self.ham_set[1].matrix
        )
        np.testing.assert_array_equal(
            np.array(
                [[-0.125, 0, 0, 0],
                 [0, 0.125, -0.5, 0],
                 [0, -0.5, 0.125, 0],
                 [0, 0, 0, -0.125]]
            ),
            self.ham_set[2].matrix
        )
        np.testing.assert_array_equal(
            np.array(
                [[-0.25, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, -0.5, 0, 0, 0, 0, 0],
                 [0, -0.5, 0.25, 0, -0.5, 0, 0, 0],
                 [0, 0, 0, 0, 0, -0.5, 0, 0],
                 [0, 0, -0.5, 0, 0, 0, 0, 0],
                 [0, 0, 0, -0.5, 0, 0.25, -0.5, 0],
                 [0, 0, 0, 0, 0, -0.5, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, -0.25]]
            ),
            self.ham_set[3].matrix
        )


class TestSpinOperators(unittest.TestCase):

    def test_SOp(self):
        spin_half_ops = SpinOperators()
        np.testing.assert_array_equal(
            np.array(
                [[0, 1],
                 [1, 0]]
            ),
            spin_half_ops.Sp + spin_half_ops.Sm
        )
        np.testing.assert_array_equal(
            np.array(
                [[0, -1j],
                 [1j, 0]]
            ),
            -1j * (spin_half_ops.Sp - spin_half_ops.Sm)
        )


if __name__ == '__main__':
    unittest.main()
