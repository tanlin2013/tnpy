import unittest
from pathlib import Path
from tnpy.finite_algorithm_base import FiniteAlgorithmBase
from xxz import XXZ


class TestFiniteAlgorithmBase(unittest.TestCase):

    model = XXZ(N=10, delta=0.5)

    def test_save(self):

        base = FiniteAlgorithmBase(self.model.mpo(), chi=20)
        base.save_mps(f'{Path(__file__).parent}/test.npz')

    def test_load(self):

        base = FiniteAlgorithmBase(
            self.model.mpo(),
            init_method=f'{Path(__file__).parent}/test.npz'
        )


if __name__ == '__main__':
    unittest.main()
