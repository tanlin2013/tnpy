import unittest
import numpy as np
from tnpy.linalg import KrylovExpm


class TestLinAlg(unittest.TestCase):

    def test_eigshmv(self):
        # self.assertEqual(True, False)
        pass


class TestKrylovExpm(unittest.TestCase):

    def test_construct_krylov_space(self):
        mat = np.random.random((50, 50))
        v0 = np.random.random(50)

        kexpm = KrylovExpm(1e-3, mat, v0)
        kexpm.construct_krylov_space()


if __name__ == '__main__':
    unittest.main()
