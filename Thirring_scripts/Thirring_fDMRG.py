import os
import argparse
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

        elem=[[I2,-0.5*Sp, -0.5*Sm, 2.0*np.sqrt(self.lamda)*Sz, self.g*Sz, gamma*I2+beta*Sz],
                    [O2,O2,O2,O2,O2,Sm],
                    [O2,O2,O2,O2,O2,Sp],
                    [O2,O2,O2,I2,O2,np.sqrt(self.lamda)*Sz],
                    [O2,O2,O2,O2,O2,Sz],
                    [O2,O2,O2,O2,O2,I2]]

        M=MPO.assign(elem,site)
        return M

if __name__=='__main__':

    parser=argparse.ArgumentParser()
    parser.add_argument("N",help="specify the system size",type=int)
    parser.add_argument("g",help="specify the coupling constant",type=float)
    parser.add_argument("ma",help="specify the mass times the lattice spacing",type=float)
    parser.add_argument("lamda",help="specify the penalty strength",type=float)
    parser.add_argument("S_target",help="specify the targeting state",type=float)
    parser.add_argument("chi",help="specify the vitual bond dimensions",type=int)
    parser.add_argument("tolerance",help="specify the tolerance",type=float)
    args=parser.parse_args()

    N=args.N
    g=args.g
    ma=args.ma
    mu=0.0
    lamda=args.lamda
    S_target=args.S_target
    chi=args.chi
    tolerance=args.tolerance

    #old_chi=500 
    whichMPS='f' ; d=2
    MPS=TNpy.tnstate.MPS(whichMPS,d,chi,N)
    Gs=MPS.initialize()
    #Gs=TNpy.data.io.read('/home/DLIN/tanlin/data/XXZ/N_{}/chi_{}/MPS-N_{}-g_{}-ma_{}-lambda_{}-Starget_{}-chi_{}-tol_{}.npz'.format(N,old_chi,N,g,ma,lamda,S_target,old_chi,tolerance))
    #Gs=TNpy.tnstate.increase_bond_dim(Gs,old_chi,chi)
    #Gs=TNpy.tnstate.lengthen_fmps(Gs,N)
    model=Thirring(N,g,ma,mu,lamda,S_target)

    simulation=TNpy.algorithm.fDMRG(model.M,Gs,N,d,chi,tolerance)
    E,stats=simulation.variational_optimize(modified_DM=True)
    Gs=simulation.Gs

    TNpy.data.io.save(os.getcwd()+'/MPS-N_{}-g_{}-ma_{}-lambda_{}-Starget_{}-chi_{}-tol_{}'.format(N,g,ma,lamda,S_target,chi,tolerance),Gs)
    Sz_i=TNpy.measurement.Sz_site(Gs)
    stag_Sz_i=TNpy.measurement.Sz_site(Gs,staggering=True)
    var=TNpy.measurement.variance(model.M,Gs)

    dE=stats['dE']; sweep=stats['sweep']; t=stats['AvgProcT']
    print "\n","="*140,"\n"
    header=["E/N","dE","variance","sum_i <Sz_i>","|sum_i <(-1)**i*Sz_i>|","sweep","AvgProcT(mins/sweep)"]
    outs=[E,dE,var,np.sum(Sz_i),abs(np.sum(stag_Sz_i)),sweep,t]
    for line in [header,outs]:
        print('{:>20} {:>20} {:>20} {:>18} {:>26} {:>8} {:>22}'.format(*line))

    print "\n","-"*140,"\n"
    header=["N","g","ma","mu","lambda","S_target","chi","tolerance"]
    outs=[N,g,ma,mu,lamda,S_target,chi,tolerance]
    for line in [header,outs]:
        print('{:>6} {:>16} {:>16} {:>10} {:>10} {:>12} {:>10} {:>12}'.format(*line))
    print "\n","="*140

    
