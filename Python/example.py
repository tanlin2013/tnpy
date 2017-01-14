import numpy as np
import Operation.MPS as MPS
import Algorithm
import Measurement
import MPO

N=  # system size
d=2  # physical bond dim
D=   # visual bond dim of MPO   
chi=  # visual bond dim of MPS

if __name__=main:

    MPS=MPS()
    Gs,SVMs=MPS.initialize_MPS()
    
    elem=[]
    M=MPO.MPO(elem)
    
    simulation=Algorithm.fDMRG()
    =simulation.variational_optimize()
    Gs=simulation.Gs()
    
    
