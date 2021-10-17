from tnpy.model import RandomHeisenberg
from tnpy.tsdrg import TSDRG


if __name__ == "__main__":

    N = 10
    h = 3.0
    penalty = 0.0
    s_target = 0
    chi = 32

    model = RandomHeisenberg(N, h, penalty, s_target)
    sdrg = TSDRG(model.mpo, chi=chi)
    sdrg.run()
    print([tree.id for tree in sdrg.tree])
    # print(sdrg.tree[50])
