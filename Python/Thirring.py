"""
This file is used for the study of Thirring model.
"""
import numpy as np
import sys ; sys.path.append("./src")
import operation
import algorithm
import measurement
import operators

class Thirring:
    def __init__(self,g,m,mu,lamda,S_target):
        self.g=g
        self.m=m
        self.mu=mu
        self.lamda=lamda
        self.S_target=Starget
        
    def M(self,site):
        MPO=operators.MPO(whichMPS,N,D)
        Sp,Sm,Sz,I2,O2=operators.spin_operators()
        
        beta=-0.5*self.J*(self.delta+eta*self.deltap)+0.5*((-1.0)**site*self.m+self.mu)-self.lamda*self.S_target
        gamma=self.lamda*(0.25+self.S_target**2/self.l)-0.25*self.J*(self.delta+eta*self.deltap)+0.5*self.mu
        parity=(site+1)%2
              
        elem=[[I2,-0.5*self.J*Sp, -0.5*self.J*Sm, 0.5*np.sqrt(self.lamda)*Sz, -0.5*self.J*(parity*self.delta+0.5*self.deltap)*Sz, gamma*I2+beta*Sz],
                    [O2,O2,O2,O2,O2,Sm],
                    [O2,O2,O2,O2,O2,Sp],
                    [O2,O2,O2,I2,O2,np.sqrt(self.lamda)*Sz],
                    [O2,O2,O2,O2,O2,Sz],
                    [O2,O2,O2,O2,O2,I2]] 
                    
        M=MPO.assign_to_MPO(elem,site)    
        return M    

if __name__=='__main__':

    whichMPS="f"
    N=10  # system size
    d=2  # physical bond dim
    D=6  # visual bond dim of MPO   
    chi=10  # visual bond dim of MPS
    g=0.5 
    m=
    mu=
    lamda=
    S_target=

    MPS=operation.MPS(whichMPS,d,chi)
    Gs=MPS.initialize_MPS(N)
    model=Thirring(g,m,mu,lamda,S_target)
    
    simulation=algorithm.fDMRG(model.M,Gs,N,d,chi,tolerance=1e-6)
    E,stats=simulation.variational_optimize()
    Gs=simulation.Gs
