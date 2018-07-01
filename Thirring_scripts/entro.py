import os,argparse
import numpy as np
import TNpy

class Thirring:
    def __init__(self,N,g,ma,mu,lamda,S_target):
        self.N=N
        self.g=g
        self.ma=ma
        self.mu=mu
        self.lamda=lamda
        self.S_target=S_target

    def M(self,site):
        MPO=TNpy.operators.MPO(whichMPS='f',D=6,N=self.N)
        Sp,Sm,Sz,I2,O2=TNpy.operators.spin()

        beta=self.g+((-1.0)**site*self.ma+self.mu)-2.0*self.lamda*self.S_target
        gamma=self.lamda*(0.25+self.S_target**2/self.N)+0.25*self.g+0.5*self.mu
        parity=0.5#(site+1)%2

        elem=[[I2,-0.5*Sp, -0.5*Sm, 2.0*np.sqrt(self.lamda)*Sz, 2.0*self.g*parity*Sz, gamma*I2+beta*Sz],
                    [O2,O2,O2,O2,O2,Sm],
                    [O2,O2,O2,O2,O2,Sp],
                    [O2,O2,O2,I2,O2,np.sqrt(self.lamda)*Sz],
                    [O2,O2,O2,O2,O2,Sz],
                    [O2,O2,O2,O2,O2,I2]]

        M=MPO.assign(elem,site)
        return M

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("N",help="specify the system size",type=int)
    parser.add_argument("g",help="specify the coupling constant",type=float)
    parser.add_argument("ma",help="specify the mass times the lattice spacing",type=float)
    parser.add_argument("chi",help="specify the vitual bond dimensions",type=int)
    args = parser.parse_args()

    N = args.N 
    g = args.g 
    ma = args.ma 
    mu = 0.0 
    lamda = 100.0 
    S_target = 0.0 
    chi = args.chi 
    tolerance = 1e-8 

    try:
        path = '/data01/mesonqcd/kcichy/thirring/scripts/MPS-N_{}-g_{}-ma_{}-lambda_{}-Starget_{}-chi_{}-tol_{}.npz'.format(N,g,ma,lamda,S_target,chi,tolerance)
        Gs = TNpy.data.io.read(path)
    except IOError:
        path = '/data01/mesonqcd/kcichy/thirring/scripts/MPS-N_{}-g_{}-ma_{}-lambda_{}-Starget_{}-chi_{}-tol_{}.npz'.format(N,g,ma,lamda,S_target,chi,5e-8)
        Gs = TNpy.data.io.read(path)

    entropy = TNpy.measurement.bipartite_entanglement_entropy(Gs[:])

    np.save(os.getcwd()+'/EE-N_{}-g_{}-ma_{}-chi_{}'.format(N,g,ma,chi),entropy)

