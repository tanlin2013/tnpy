from tnpy.model import DimerXXZ
from tnpy.tsdrg import TSDRG


if __name__ == "__main__":

    N = 32
    J = 0.1
    delta = 0.1
    h = 0.0
    penalty = 0.0
    s_target = 0
    chi = 8

    model = DimerXXZ(N, J, delta, h, penalty, s_target)
    tsdrg = TSDRG(model.mpo, chi=chi)
    tsdrg.run()
    print([node.id for node in tsdrg.tree])
    # print(sdrg.tree[50])
