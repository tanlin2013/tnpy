import numpy as np
import sys ; sys.path.append("./src")
import Operation
import Algorithm
import Measurement
import MPO

if __name__=='__main__':

    whichMPS="f"
    N=10  # system size
    d=2  # physical bond dim
    D=3   # visual bond dim of MPO   
    chi=10  # visual bond dim of MPS

    MPS=Operation.MPS(whichMPS,d,chi)
    Gs=MPS.initialize_MPS(N)
    
    MPO=MPO.MPO(whichMPS,N,D)
    Sp=np.array([[0,1],[0,0]],dtype=float)
    Sm=np.array([[0,0],[1,0]],dtype=float)
    Sz=0.5*np.array([[1,0],[0,-1]],dtype=float)
    I2=np.identity(2,dtype=float)
    O2=np.zeros((2,2),dtype=float)

    elem=[[Sp,Sm,I2],
          [O2,O2,Sm],
          [O2,O2,Sp]]
    MPO.elem=elem    
    M=MPO.MPO()

    simulation=Algorithm.fDMRG(M,Gs,N,d,chi)
    E,stats=simulation.variational_optimize()
    Gs=simulation.Gs()



    
    
