"""
This file contains the physical quantities to be measured.
"""

import numpy as np
import scipy.linalg as scl
import operation
import operators

global Sp,Sm,Sz,I2,O2=operators.spin_operators()
    
def variance(M,Gs,SVMs=None):
    
    return
  
def entanglement_entropy(S):
    ss=np.square(S)        
    entropy=-np.trace(ss*scl.logm(ss))
    return entropy

def Sz_site(Gs,order,staggered=False):
    N=len(Gs) ; d=2 ; state=[]
    if staggered:
        stag=-1
    else:
        stag=1
    if order=='R': # state is right-normalized
        for site in xrange(N-1):
            if site==0:                
                I=np.identity(Gs[site].shape[1],dtype=float)
                Sz_site=np.tensordot(Gs[site],np.tensordot(stag**site*Sz,np.conjugate(Gs[site]),axes=(1,0)),axes=(0,0))
                state.append(np.tensordot(Sz_site,I,axes=([0,1],[0,1])).item())
                theta=Gs[site]
            else:
                IL=np.identity(Gs[site].shape[0],dtype=float)
                IR=np.identity(Gs[site].shape[2],dtype=float)
                Sz_site=np.tensordot(Gs[site],np.tensordot(stag**site*Sz,np.conjugate(Gs[site]),axes=(1,1)),axes=(1,0))
                Sz_site=np.swapaxes(Sz_site,1,2)                
                state.append(np.tensordot(IL,np.tensordot(Sz_site,IR,axes=([2,3],[0,1])),axes=([0,1],[0,1])).item())
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
        I=np.identity(Gs[N-1].shape[1],dtype=float)
        Sz_site=np.tensordot(Gs[N-1],np.tensordot(stag**site*Sz,np.conjugate(Gs[N-1]),axes=(1,0)),axes=(0,0))
        state.append(np.tensordot(Sz_site,I,axes=([0,1],[0,1])).item())
    else: # state is left-normalized
        for site in xrange(N-1,0,-1):
            if site==N-1:                
                I=np.identity(Gs[site].shape[1],dtype=float)
                Sz_site=np.tensordot(Gs[site],np.tensordot(stag**site*Sz,np.conjugate(Gs[site]),axes=(1,0)),axes=(0,0))
                state.append(np.tensordot(I,Sz_site,axes=([0,1],[0,1])).item())
                theta=Gs[site]
            else:
                IL=np.identity(Gs[site].shape[0],dtype=float)
                IR=np.identity(Gs[site].shape[2],dtype=float)
                Sz_site=np.tensordot(Gs[site],np.tensordot(stag**site*Sz,np.conjugate(Gs[site]),axes=(1,1)),axes=(1,0))
                Sz_site=np.swapaxes(Sz_site,1,2)                
                state.append(np.tensordot(IL,np.tensordot(Sz_site,IR,axes=([2,3],[0,1])),axes=([0,1],[0,1])).item())
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
        I=np.identity(Gs[0].shape[1],dtype=float)
        Sz_site=np.tensordot(Gs[0],np.tensordot(stag**site*Sz,np.conjugate(Gs[0]),axes=(1,0)),axes=(0,0))
        state.append(np.tensordot(I,Sz_site,axes=([0,1],[0,1])).item())        
    return state      


