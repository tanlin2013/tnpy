"""
This file is used for the study of Thirring model.
"""
#import os,argparse
import numpy as np
import TNpy
import matplotlib.pyplot as plt

class Thirring:
    def __init__(self,N,g,ma,mu,lamda,S_target):
        self.N=N
        self.g=g
        self.ma=ma
        self.mu=mu
        self.lamda=lamda
        self.S_target=S_target
        
    def M(self,site):
        MPO=TNpy.operators.MPO(whichMPS='f',N=self.N,D=6)
        Sp,Sm,Sz,I2,O2=TNpy.operators.spin_operators()
        
        beta=self.g+((-1.0)**site*self.ma+self.mu)-2.0*self.lamda*self.S_target
        gamma=self.lamda*(0.25+self.S_target**2/self.N)+0.25*self.g+0.5*self.mu
        parity=(site+1)%2

        elem=[[I2,-0.5*Sp, -0.5*Sm, 2.0*np.sqrt(self.lamda)*Sz, 2.0*self.g*parity*Sz, gamma*I2+beta*Sz],
                    [O2,O2,O2,O2,O2,Sm],
                    [O2,O2,O2,O2,O2,Sp],
                    [O2,O2,O2,I2,O2,np.sqrt(self.lamda)*Sz],
                    [O2,O2,O2,O2,O2,Sz],
                    [O2,O2,O2,O2,O2,I2]]
                    
        M=MPO.assign_to_MPO(elem,site)    
        return M    

if __name__=='__main__':

    N=40  # system size  
    g=-0.5 # coupling constant
    ma=0.1 # bare mass times the lattice spacing
    mu=0.0 # chemical potential
    lamda=2000.0 # penalty strength
    S_target=0.0 # targeting state
    chi=100  # visual bond dim of MPS
    tolerance=1e-12
    
    whichMPS='f' ; d=2
    MPS=TNpy.tnstate.MPS(whichMPS,d,chi,N)
    Gs=MPS.initialize()
    model=Thirring(N,g,ma,mu,lamda,S_target)
    
    simulation=TNpy.algorithm.fDMRG(model.M,Gs,N,d,chi,tolerance)
    E,stats=simulation.variational_optimize()
    Gs=simulation.Gs
    
    #TNpy.data_analysis.iostate.save(os.getcwd()+'/MPS-N_{}-g_{}-ma_{}-lambda_{}-Starget_{}-chi_{}-tol_{}'.format(N,g,ma,lamda,S_target,chi,tolerance),Gs)
    Sz_i=TNpy.measurement.Sz_site(Gs)
    stag_Sz_i=TNpy.measurement.Sz_site(Gs,staggering=True)
    var=TNpy.measurement.variance(model.M,Gs)
    sector=sum(Sz_i); ccs=abs(sum(stag_Sz_i))/N
    
    BKT_corr=TNpy.measurement.BKT_corr(Gs[:],discard_site=2)
    dist,corr=BKT_corr.avg_corr()
    
    plt.figure(figsize=(8,6))
    plt.plot(dist,corr/abs(corr[0]),marker='o',linestyle='-')
    #plt.title('BKT correlator')
    plt.xlabel('r')
    plt.ylabel('G(r)/|G(0)|')
    plt.grid()
    #plt.xlim(-1,dist[-1]+1)
    #plt.ylim(-0.1,1.1)
    #plt.annotate('G(0) = {}'.format(corr[0]), xy=(0.15,0.85), xycoords='axes fraction')
    #plt.figtext(0.085,0.05,'')
    #plt.legend(('discard 1','discard 2','discard 3','discard 4',),loc=1,numpoints=1) 
    #plt.savefig('/home/davidtan/Desktop/penalty_0/BKT_corr-N_{}-g_{}-ma_{}-lambda_{}-Starget_{}-chi_{}.eps'.format(N,g,ma,lamda,S_target,chi))
    #plt.close()    
    plt.show()
