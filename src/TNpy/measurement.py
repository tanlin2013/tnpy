"""
This file contains the physical quantities to be measured.
"""

import warnings
import numpy as np
from scipy.linalg import logm
import operators,tnstate 

class _update_Env:
    def __init__(self,MPO,Gs,N):
        self.MPO=MPO 
        self.Gs=Gs
        self.N=N      
    
    def _update_EnvL(self,EnvL,site):
        M=self.MPO(site)
        if site==self.N-1:
            EnvL=np.tensordot(np.tensordot(np.tensordot(EnvL,self.Gs[site],axes=(0,1)),
                 M,axes=([0,2],[1,0])),np.conjugate(self.Gs[site]),axes=([0,1],[1,0]))
        else:
            EnvL=np.tensordot(np.tensordot(np.tensordot(EnvL,self.Gs[site],axes=(0,0)),
                 M,axes=([0,2],[1,0])),np.conjugate(self.Gs[site]),axes=([0,2],[0,1]))
        return EnvL

    def _update_EnvR(self,EnvR,site):
        M=self.MPO(site)
        if site==0: 
            EnvR=np.tensordot(np.tensordot(np.tensordot(EnvR,self.Gs[site],axes=(0,1)),
                 M,axes=([0,2],[1,0])),np.conjugate(self.Gs[site]),axes=([0,1],[1,0]))   
        else:
            EnvR=np.tensordot(np.tensordot(np.tensordot(EnvR,self.Gs[site],axes=(0,2)),
                 M,axes=([0,3],[3,0])),np.conjugate(self.Gs[site]),axes=([0,3],[2,1]))
        return EnvR 
    
    def _update_EnvL2(self,EnvL2,site):
        M=self.MPO(site)
        if site==self.N-1:
            EnvL2=np.tensordot(np.tensordot(np.tensordot(np.tensordot(EnvL2,
                  self.Gs[site],axes=(0,1)),M,axes=([0,3],[1,0])),
                  M,axes=([0,2],[1,2])),np.conjugate(self.Gs[site]),axes=([0,1],[1,0]))
        else:
            EnvL2=np.tensordot(np.tensordot(np.tensordot(np.tensordot(EnvL2,
                  self.Gs[site],axes=(0,0)),M,axes=([0,3],[1,0])),
                  M,axes=([0,3],[1,2])),np.conjugate(self.Gs[site]),axes=([0,3],[0,1]))
        return EnvL2

    def _update_EnvR2(self,EnvR2,site):
        M=self.MPO(site)
        if site==0:    
            EnvR2=np.tensordot(np.tensordot(np.tensordot(np.tensordot(EnvR2,
                  self.Gs[site],axes=(0,1)),M,axes=([0,3],[1,0])),
                  M,axes=([0,2],[1,2])),np.conjugate(self.Gs[site]),axes=([0,1],[1,0]))
        else:
            EnvR2=np.tensordot(np.tensordot(np.tensordot(np.tensordot(EnvR2,
                  self.Gs[site],axes=(0,2)),M,axes=([0,4],[3,0])),
                  M,axes=([0,4],[3,2])),np.conjugate(self.Gs[site]),axes=([0,3],[2,1]))
        return EnvR2

def variance(MPO,Gs):
    N=len(Gs); h=_update_Env(MPO,Gs,N); order=tnstate.get_mps_order(Gs)
    if order=='L':
        for site in xrange(N):
            if site==0:
                M=MPO(site)
                EnvL=tnstate.transfer_operator(Gs[site],M)
                EnvL2=np.tensordot(np.tensordot(np.tensordot(Gs[site],
                      M,axes=(0,0)),M,axes=(2,2)),np.conjugate(Gs[site]),axes=(2,0))
            else:
                EnvL=h._update_EnvL(EnvL,site)
                EnvL2=h._update_EnvL2(EnvL2,site)
        var=EnvL2.item()-EnvL.item()**2    
    elif order=='R':
        for site in xrange(N-1,-1,-1):
            if site==N-1:
                M=MPO(site)
                EnvR=tnstate.transfer_operator(Gs[site],M)
                EnvR2=np.tensordot(np.tensordot(np.tensordot(Gs[site],
                      M,axes=(0,0)),M,axes=(2,2)),np.conjugate(Gs[site]),axes=(2,0))
            else:
                EnvR=h._update_EnvR(EnvR,site)
                EnvR2=h._update_EnvR2(EnvR2,site)
        var=EnvR2.item()-EnvR.item()**2       
    if var < 0.0:
        warnings.simplefilter("always")
        warnings.warn("PrecisionError: encounter negative variance after the subtraction.")
    return var

def entanglement_entropy(S):
    ss=np.square(S)        
    entropy=-np.trace(ss*logm(ss))
    return entropy

def Sz_site(Gs,spin=0.5,staggering=False):
    Sp,Sm,Sz,I2,O2=operators.spin_operators(spin)
    N=len(Gs); order=tnstate.get_mps_order(Gs); d=1./spin; state=[None]*N
    if staggering: stag=-1
    else: stag=1
    
    def update_Sz(site):
        if site==0 or site==N-1:
            I=np.identity(Gs[site].shape[1],dtype=float)
            Sz_site=np.tensordot(np.tensordot(np.tensordot(I,Gs[site],axes=(0,1)),
                    stag**site*Sz,axes=(1,0)),np.conjugate(Gs[site]),axes=([0,1],[1,0]))
        else:
            IL=np.identity(Gs[site].shape[0],dtype=float)
            IR=np.identity(Gs[site].shape[2],dtype=float)
            Sz_site=np.tensordot(np.tensordot(np.tensordot(np.tensordot(IL,
                    Gs[site],axes=(0,0)),stag**site*Sz,axes=(1,0)),
                    np.conjugate(Gs[site]),axes=([0,2],[0,1])),IR,axes=([0,1],[0,1]))
        return Sz_site.item() 
        
    if order=='R': # state is right-normalized
        for site in xrange(N):
            state[site]=update_Sz(site)
            if site < N-1:
                if site==0:    
                    theta=Gs[site]
                else:
                    theta=np.ndarray.reshape(Gs[site],(d*Gs[site].shape[0],Gs[site].shape[2]))     
                X,S,Y=np.linalg.svd(theta,full_matrices=False)                
                if site==N-2:
                    Gs[site+1]=np.tensordot(Gs[site+1],np.dot(np.diagflat(S),Y),axes=(1,1))
                else:
                    Gs[site+1]=np.tensordot(np.dot(np.diagflat(S),Y),Gs[site+1],axes=(1,0))
                if site==0:
                    Gs[site]=np.ndarray.reshape(X,(d,Gs[site].shape[1]))
                else:
                    Gs[site]=np.ndarray.reshape(X,(Gs[site].shape[0],d,Gs[site].shape[2]))
    elif order=='L': # state is left-normalized
        for site in xrange(N-1,-1,-1):
            state[site]=update_Sz(site)
            if site > 0:
                if site==N-1:      
                    theta=Gs[site]
                else:    
                    theta=np.ndarray.reshape(Gs[site],(Gs[site].shape[0],d*Gs[site].shape[2]))     
                X,S,Y=np.linalg.svd(theta,full_matrices=False)                
                if site==1:
                    Gs[site-1]=np.tensordot(Gs[site-1],np.dot(X,np.diagflat(S)),axes=(1,0))
                else:         
                    Gs[site-1]=np.tensordot(Gs[site-1],np.dot(X,np.diagflat(S)),axes=(2,0))
                if site==N-1:
                    Gs[site]=np.ndarray.reshape(Y,(d,Gs[site].shape[1]))
                else:
                    Gs[site]=np.ndarray.reshape(Y,(Gs[site].shape[0],d,Gs[site].shape[2]))
    return state 

"""
def correlation_function(Gs,m,n):
    Sp,Sm,Sz,I2,O2=operators.spin_operators()
    order=tnstate.get_mps_order(Gs)
    
    return correlator
"""
