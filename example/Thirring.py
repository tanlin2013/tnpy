"""
This file is used for the study of Thirring model.
"""
import numpy as np
import TNpy

class Thirring:
    def __init__(self,N,a,g,m,mu,lamda,S_target):
        self.N=N
        self.J=1/a
        self.g=g
        self.m=m
        self.mu=mu
        self.lamda=lamda
        self.S_target=S_target
        
    def M(self,site):
        MPO=TNpy.operators.MPO(whichMPS='f',N=self.N,D=6)
        Sp,Sm,Sz,I2,O2=TNpy.operators.spin_operators()
        
        beta=self.J*self.g+((-1.0)**site*self.m+self.mu)-2.0*self.lamda*self.S_target
        gamma=self.lamda*(0.25+self.S_target**2/self.N)+0.25*self.J*self.g+0.5*self.mu
        parity=(site+1)%2
              
        elem=[[I2,-0.5*self.J*Sp, -0.5*self.J*Sm, 2.0*np.sqrt(self.lamda)*Sz, 2.0*self.J*self.g*parity*Sz, gamma*I2+beta*Sz],
                    [O2,O2,O2,O2,O2,Sm],
                    [O2,O2,O2,O2,O2,Sp],
                    [O2,O2,O2,I2,O2,np.sqrt(self.lamda)*Sz],
                    [O2,O2,O2,O2,O2,Sz],
                    [O2,O2,O2,O2,O2,I2]] 
                    
        M=MPO.assign_to_MPO(elem,site)    
        return M    

if __name__=='__main__':

    N=40  # system size  
    chi=20  # visual bond dim of MPS
    a=1.0 # lattice spacing
    g=0.5 # coupling constant
    m=0.1 # bare mass
    mu=0.0 # chemical potential
    lamda=0.0 # penalty strength
    S_target=0.0

    MPS=TNpy.tnstate.MPS(whichMPS='f',d=2,**chi,**N)
    Gs=MPS.initialize()
    model=Thirring(N,a,g,m,mu,lamda,S_target)
    
    simulation=TNpy.algorithm.fDMRG(model.M,Gs,N,d,chi,tolerance=1e-8)
    E,stats=simulation.variational_optimize()
    Gs=simulation.Gs
