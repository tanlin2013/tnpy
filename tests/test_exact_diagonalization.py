from unittest import TestCase
import numpy as np
from tnpy.model import RandomHeisenberg
from tnpy.exact_diagonalization import ExactDiagonalization


class TestExactDiagonalization(TestCase):

    def __init__(self, *args, **kwargs):
        super(TestExactDiagonalization, self).__init__(*args, **kwargs)
        self.ed = ExactDiagonalization(RandomHeisenberg(n=2, h=0).mpo)

    def test_matrix(self):
        np.testing.assert_array_equal(
            np.array(
                [[0.25, 0, 0, 0],
                 [0, -0.25, 0.5, 0],
                 [0, 0.5, -0.25, 0],
                 [0, 0, 0, 0.25]]
            ),
            self.ed.matrix
        )

    def test_evals(self):
        np.testing.assert_array_equal(
            np.array([-0.75, 0.25, 0.25, 0.25]),
            self.ed.evals
        )

    def test_evecs(self):
        np.testing.assert_array_equal(
            np.array(
                [[0, 1, 0, 0],
                 [1/np.sqrt(2), 0, 1/np.sqrt(2), 0],
                 [-1/np.sqrt(2), 0, 1/np.sqrt(2), 0],
                 [0, 0, 0, 1]]
            ),
            self.ed.evecs
        )
