import numpy as np
import sys ; sys.path.append("./src")
import operation
import algorithm
import measurement
import operators

if __name__=='__main__':

    whichMPS="f"
    N=10  # system size
    d=2  # physical bond dim
    D=3   # visual bond dim of MPO   
    chi=10  # visual bond dim of MPS

    MPS=operation.MPS(whichMPS,d,chi)
    Gs=MPS.initialize_MPS(N)
    
    def M(site):
        MPO=operators.MPO(whichMPS,N,D)
        Sp,Sm,Sz,I2,O2=operators.spin_operators()

        elem=[[Sp,Sm,I2],
              [O2,O2,Sm],
              [O2,O2,Sp]]
        M=MPO.assign_to_MPO(elem,site)    
        return M

    simulation=algorithm.fDMRG(M,Gs,N,d,chi)
    E,stats=simulation.variational_optimize()
    Gs=simulation.Gs()



    
    
