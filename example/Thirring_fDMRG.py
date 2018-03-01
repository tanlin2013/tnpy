"""
This file is used for the study of Thirring model.
"""
#import os,argparse
import numpy as np
import TNpy

class Thirring:
    def __init__(self, N, g, ma, mu, lamda, S_target):
        self.N = N
        self.g = g
        self.ma = ma
        self.mu = mu
        self.lamda = lamda
        self.S_target = S_target
        
    def M(self, site):
        MPO = TNpy.operators.MPO(whichMPS='f', D=6, N=self.N)
        Sp, Sm, Sz, I2, O2 = TNpy.operators.spin()
        
        beta = self.g + ((-1.0)**site*self.ma+self.mu) - 2.0*self.lamda*self.S_target
        gamma = self.lamda*(0.25 + self.S_target**2/self.N) + 0.25*self.g + 0.5*self.mu

        elem = [[I2, -0.5*Sp, -0.5*Sm, 2.0*np.sqrt(self.lamda)*Sz, self.g*parity*Sz, gamma*I2+beta*Sz],
                    [O2, O2, O2, O2, O2, Sm],
                    [O2, O2, O2, O2, O2, Sp],
                    [O2, O2, O2, I2, O2, np.sqrt(self.lamda)*Sz],
                    [O2, O2, O2, O2, O2, Sz],
                    [O2, O2, O2, O2, O2, I2]]
                    
        M = MPO.assign(elem,site)    
        return M    

if __name__ == '__main__':

    N = 40  # system size  
    g = -0.5 # coupling constant
    ma = 0.1 # bare mass times the lattice spacing
    mu = 0.0 # chemical potential
    lamda = 2000.0 # penalty strength
    S_target = 0.0 # targeting state
    chi = 100  # visual bond dim of MPS
    tolerance = 1e-12
    
    whichMPS = 'f'; d = 2
    MPS = TNpy.tnstate.MPS(whichMPS,d,chi,N)
    Gs = MPS.initialize()
    model = Thirring(N,g,ma,mu,lamda,S_target)
    
    simulation = TNpy.algorithm.fDMRG(model.M,Gs,N,d,chi,tolerance)
    E, stats = simulation.variational_optimize()
    Gs = simulation.Gs
    
    #TNpy.data.io.save(os.getcwd()+'/MPS-N_{}-g_{}-ma_{}-lambda_{}-Starget_{}-chi_{}-tol_{}'.format(N,g,ma,lamda,S_target,chi,tolerance),Gs)
    Sz_i = TNpy.measurement.Sz_site(Gs)
    stag_Sz_i = TNpy.measurement.Sz_site(Gs,staggering=True)
    var = TNpy.measurement.variance(model.M,Gs)
    sector = sum(Sz_i); ccs = abs(sum(stag_Sz_i))/N
    
    BKT_corr = TNpy.measurement.BKT_corr(Gs[:],discard_site=2)
    dist, corr = BKT_corr.avg_corr()
    
