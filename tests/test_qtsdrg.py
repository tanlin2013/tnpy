from unittest import TestCase
import numpy as np
from quimb.tensor.tensor_1d import MatrixProductOperator as MPO
from tnpy.model import RandomHeisenberg
from tnpy.exact_diagonalization import ExactDiagonalization as ED
from tnpy.qtsdrg import (
    TensorTree,
    TreeTensorNetworkSDRG as tSDRG,
    TreeTensorNetworkMeasurements
)


class TestTensorTree(TestCase):

    def __init__(self, *args, **kwargs):
        super(TestTensorTree, self).__init__(*args, **kwargs)
        model = RandomHeisenberg(N=4, h=0.0, penalty=0.0, s_target=0)
        mpo = MPO([mpo.tensor for mpo in model.mpo])
        self.tree = TensorTree(mpo)
        self.tree.fuse(left_id=1, right_id=2, new_id=4,
                       data=ED(RandomHeisenberg(N=2, h=0).mpo).evecs.reshape((2, 2, 4)))
        self.tree.fuse(left_id=0, right_id=4, new_id=5,
                       data=ED(RandomHeisenberg(N=3, h=0).mpo).evecs.reshape((2, 4, 8)))
        self.tree.fuse(left_id=5, right_id=3, new_id=6,
                       data=ED(RandomHeisenberg(N=4, h=0).mpo).evecs.reshape((8, 2, 16)))

    def test_find_path(self):
        self.tree.find_path(1)

    def test_tensor_network(self):
        net = self.tree.tensor_network(conj=True)
        # net.graph()


class TestTreeTensorNetworkSDRG(TestCase):

    def __init__(self, *args, **kwargs):
        super(TestTreeTensorNetworkSDRG, self).__init__(*args, **kwargs)
        model = RandomHeisenberg(N=6, h=10.0, penalty=0.0, s_target=0)
        mpo = MPO([mpo.tensor for mpo in model.mpo])
        self.tsdrg = tSDRG(mpo, chi=2 ** 6)
        self.tsdrg.run()
        self.ed = ED(model.mpo)

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
        np.testing.assert_allclose(
            self.ed.evals[:self.tsdrg.chi],
            self.tsdrg.evals,
            atol=1e-12
        )

    def test_measurements(self):
        print(self.tsdrg.measurements.sandwich())


class TestTreeTensorNetworkMeasurements(TestCase):

    def __init__(self, *args, **kwargs):
        super(TestTreeTensorNetworkMeasurements, self).__init__(*args, **kwargs)
        model = RandomHeisenberg(N=20, h=10.0, penalty=0.0, s_target=0)
        self.mpo = MPO([mpo.tensor for mpo in model.mpo])
        self.tsdrg = tSDRG(self.mpo, chi=2 ** 6)
        self.tsdrg.run()

    def test_sandwich(self):
        func = TreeTensorNetworkMeasurements(self.tsdrg.tree)
        np.testing.assert_allclose(
            self.tsdrg.evals,
            np.diag(func.sandwich(self.mpo).data),
            atol=1e-12
        )
        np.testing.assert_allclose(
            np.identity(2**6),
            func.sandwich().data,
            atol=1e-12
        )
