from unittest import TestCase
import numpy as np
from tnpy.operators import SpinOperators, FullHamiltonian
from tnpy.model import XXZ, RandomHeisenberg


class TestSpinOperators(TestCase):

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


class TestMatrixProductOperator(TestCase):

    def test_square(self):
        bilayer_mpo = RandomHeisenberg(n=4, h=0).mpo.square()
        self.assertCountEqual(
            (36, 2, 2),
            bilayer_mpo[0].shape
        )
        self.assertCountEqual(
            (36, 36, 2, 2),
            bilayer_mpo[1].shape
        )
        self.assertCountEqual(
            (36, 36, 2, 2),
            bilayer_mpo[2].shape
        )
        self.assertCountEqual(
            (36, 2, 2),
            bilayer_mpo[3].shape
        )


class TestFullHamiltonian(TestCase):

    def __init__(self, *args, **kwargs):
        super(TestFullHamiltonian, self).__init__(*args, **kwargs)
        self.ham1 = FullHamiltonian(RandomHeisenberg(n=2, h=0).mpo)
        self.ham2 = FullHamiltonian(RandomHeisenberg(n=3, h=0).mpo)
        self.ham3 = FullHamiltonian(XXZ(n=2, delta=0.5).mpo)
        self.ham4 = FullHamiltonian(XXZ(n=3, delta=0.5).mpo)

    def test_n_sites(self):
        self.assertEqual(2, self.ham1.n_sites)
        self.assertEqual(3, self.ham2.n_sites)
        self.assertEqual(2, self.ham3.n_sites)
        self.assertEqual(3, self.ham4.n_sites)

    def test_matrix(self):
        np.testing.assert_array_equal(
            np.array(
                [[0.25, 0, 0, 0],
                 [0, -0.25, 0.5, 0],
                 [0, 0.5, -0.25, 0],
                 [0, 0, 0, 0.25]]
            ),
            self.ham1.matrix
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
            self.ham2.matrix
        )
        np.testing.assert_array_equal(
            np.array(
                [[-0.125, 0, 0, 0],
                 [0, 0.125, -0.5, 0],
                 [0, -0.5, 0.125, 0],
                 [0, 0, 0, -0.125]]
            ),
            self.ham3.matrix
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
            self.ham4.matrix
        )
