import unittest
from tnpy.operators import MPO
from xxz import XXZ


class TestMPO(unittest.TestCase):

    model = XXZ(N=10, delta=0.5)

    def test_mpo(self):

        mpo = self.model.mpo()
        print(mpo.nodes)


if __name__ == '__main__':
    unittest.main()
