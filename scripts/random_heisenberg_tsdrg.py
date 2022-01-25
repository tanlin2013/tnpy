import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tnpy.model import RandomHeisenberg
from tnpy.tsdrg import TreeTensorNetworkSDRG as tSDRG
from tnpy.exact_diagonalization import ExactDiagonalization as ED

np.set_printoptions(threshold=sys.maxsize)


if __name__ == "__main__":

    model = RandomHeisenberg(n=6, h=0.1, penalty=0, s_target=0, seed=2021)
    tsdrg = tSDRG(model.mpo, chi=2**4)
    tsdrg.run()
    # ed = ED(model.mpo)

    # with open('test.pickle', 'rb') as f:
    #     tsdrg = pickle.load(f)

    # evals = np.diag(tsdrg.energies(model.mpo))
    # idx = np.argsort(evals)

    # cov = tsdrg.energies(model2.mpo) - np.square(tsdrg.energies(model.mpo))
    # print(cov)
    # print(ed.evals)
    # print(evals[idx])
    # print(np.sum(cov[idx], axis=1))
    tsdrg.tree.plot().render(format='png', view=True)

    # with open('test.pickle', 'wb') as f:
    #     pickle.dump(tsdrg, f)

    # plt.scatter(ed.evals, [5] * len(ed.evals))
    # plt.scatter(evals, [4] * len(evals))
    # plt.ylim(0, 10)
    # plt.show()
