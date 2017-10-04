"""
This file contains the fundamental functions for the Matrix Product State (MPS) operations. 
"""

import numpy as np

class MPS:   
    def __init__(self,whichMPS,d,chi,N=None):
        """
        * Parameters:
            * whichMPS: string, {'i','f'} 
                If whichMPS='i', an infinite MPS is initialized. Otherwise if whichMPS='f', a finite MPS is created.  
            * d: int
                The physical bond dimension which is associated to the dimension of single-particle Hilbert space.
            * chi: int
                The visual bond dimension to be keep after the Singular Value Decomposition (SVD).
            * N: int, optional
                If whichMPS='f', the size of system N is needed.     
        """
        if whichMPS!='i' and whichMPS!='f':
            raise ValueError('Only iMPS and fMPS are supported.')
        self.whichMPS=whichMPS
        self.d=d
        self.chi=chi
        self.N=N
        
    def initialize(self,canonical_form='R'):
        """
        Randomly initialize the MPS.
    
        * Parameters:                     
            * canonical_form: string, {'L','R'}, default='R'
                If whichMPS='f', fMPS can be represented as left-normalized or right-normalized MPS.            
        
        * Returns: 
            * Gs: list of ndarray
                A list of rank-3 tensors which represents the MPS. The order of tensor is (chi,d,chi) or (d,chi) for the boundaries if fMPS is considered.  
            * SVMs: list of ndarray
                A list of singular value matrices. SVMs is always return for iMPS. But, for the fMPS SVMs only return when canonical_form='GL'.
        """
        
        # Check the input variables
        if self.whichMPS=='f': 
            if not canonical_form in ['L','R'] or type(self.N) is not int:
                raise ValueError('canonical_form and size must be specified when whichMPS=f.')        
        
        if self.whichMPS=='i':
            # Create the iMPS
            Gs=[None]*2 ; SVMs=[None]*2
            for site in xrange(2):
                Gs[site]=np.random.rand(self.chi,self.d,self.chi)
                SVMs[site]=np.diagflat(np.random.rand(self.chi))
            return Gs,SVMs    
        elif self.whichMPS=='f':
            # Create the fMPS
            Gs=[None]*self.N ; N_parity=self.N%2
            for site in xrange(self.N):        
                if N_parity==0:
                    if site==0 or site==self.N-1:
                        Gs[site]=np.random.rand(self.d,min(self.d,self.chi))
                    elif site<=self.N/2-1 and site!=0:
                        Gs[site]=np.random.rand(min(self.d**site,self.chi),self.d,min(self.d**(site+1),self.chi))
                    elif site>self.N/2-1 and site!=self.N-1:
                        Gs[site]=np.random.rand(min(self.d**(self.N-site),self.chi),self.d,min(self.d**(self.N-1-site),self.chi))
                elif N_parity==1:
                    if site==0 or site==self.N-1:
                        Gs[site]=np.random.rand(self.d,min(self.d,self.chi))
                    elif site<self.N/2 and site!=0:
                        Gs[site]=np.random.rand(min(self.d**site,self.chi),self.d,min(self.d**(site+1),self.chi))
                    elif site==self.N/2:
                        Gs[site]=np.random.rand(min(self.d**site,self.chi),self.d,min(self.d**site,self.chi))
                    elif site>self.N/2 and site!=self.N-1:
                        Gs[site]=np.random.rand(min(self.d**(self.N-site),self.chi),self.d,min(self.d**(self.N-1-site),self.chi))        
            # Canonical normalization of the MPS
            if canonical_form=='L':
                Gs=normalize_fMPS(Gs,order='L')
            elif canonical_form=='R':
                Gs=normalize_fMPS(Gs,order='R')
            return Gs       
    
def normalize_fMPS(Gs,order):
    """
    Canonical normalization of the fMPS.
        
    * Parameters:
        * Gs: list of ndarray
            The fMPS wants to be left- or right-normalized.  
        * order: string, {'L','R','GL'}
            Specified the direction of normalization.
    * Returns:
        * Gs: list of ndarray
            Left- or right-normalized MPS.
    """
    N=len(Gs); d=Gs[0].shape[0]
    if order=='L':
        for site in xrange(N-1):
            if site==0:
                theta=Gs[site]
            else:
                theta=np.ndarray.reshape(Gs[site],(d*Gs[site].shape[0],Gs[site].shape[2]))
            X,S,Y=np.linalg.svd(theta,full_matrices=False)
            if site==N-2:
                Gs[site+1]=np.tensordot(Gs[site+1],np.dot(np.diagflat(S/np.linalg.norm(S)),Y),axes=(1,1))
            else:
                Gs[site+1]=np.tensordot(np.dot(np.diagflat(S/np.linalg.norm(S)),Y),Gs[site+1],axes=(1,0))
            if site==0:
                Gs[site]=np.ndarray.reshape(X,(d,Gs[site].shape[1]))
            else:
                Gs[site]=np.ndarray.reshape(X,(Gs[site].shape[0],d,Gs[site].shape[2]))
        return Gs
    elif order=='R':
        for site in range(N-1,0,-1):
            if site==N-1:
                theta=Gs[site]
            else:
                theta=np.ndarray.reshape(Gs[site],(Gs[site].shape[0],d*Gs[site].shape[2]))      
            X,S,Y=np.linalg.svd(theta,full_matrices=False)                
            if site==1:
                Gs[site-1]=np.tensordot(Gs[site-1],np.dot(X,np.diagflat(S/np.linalg.norm(S))),axes=(1,0))
            else:         
                Gs[site-1]=np.tensordot(Gs[site-1],np.dot(X,np.diagflat(S/np.linalg.norm(S))),axes=(2,0))
            if site==N-1:
                Gs[site]=np.ndarray.reshape(Y,(d,Gs[site].shape[1]))
            else:
                Gs[site]=np.ndarray.reshape(Y,(Gs[site].shape[0],d,Gs[site].shape[2]))  
        return Gs
    #elif order=='GL':
    #    return Gs,SVMs
    else:
        raise ValueError('The order must be either L or R.')

def get_mps_order(Gs):
    N=len(Gs) ; chi=Gs[0].shape[1]
    def G2(site):
        G2=np.tensordot(Gs[site],np.conjugate(Gs[site]),axes=(0,0))
        return G2
    if np.allclose(G2(0),np.identity(chi),atol=1e-12): order='L'
    elif np.allclose(G2(N-1),np.identity(chi),atol=1e-12): order='R'    
    return order        
        
def transfer_operator(G,M):
    if G.ndim==2:
        trans=np.tensordot(G,np.tensordot(M,np.conjugate(G),axes=(2,0)),axes=(0,0))
    else:                        
        trans=np.tensordot(G,np.tensordot(M,np.conjugate(G),axes=(2,1)),axes=(1,0))
        trans=np.swapaxes(trans,1,2)
        trans=np.swapaxes(trans,3,4)
        trans=np.swapaxes(trans,2,3)
    return trans    

def increase_mps_dim(Gs,old_chi,new_chi):
    N=len(Gs); d=Gs[0].shape[0]
    new_Gs=[None]*N
    for site in xrange(N):
        if site==0 or site==N-1:
            I=np.eye(min(d,old_chi),min(d,new_chi))
            new_Gs[site]=np.tensordot(Gs[site],I,axes=(1,0))
        elif site<=N/2-1 and site!=0:
            IL=np.eye(min(d**site,old_chi),min(d**site,new_chi))
            IR=np.eye(min(d**(site+1),old_chi),min(d**(site+1),new_chi))
            new_Gs[site]=np.tensordot(np.tensordot(IL,Gs[site],axes=(0,0)),IR,axes=(2,0))
        elif site>N/2-1 and site!=N-1:
            IL=np.eye(min(d**(N-site),old_chi),min(d**(N-site),new_chi))
            IR=np.eye(min(d**(N-1-site),old_chi),min(d**(N-1-site),new_chi))
            new_Gs[site]=np.tensordot(np.tensordot(IL,Gs[site],axes=(0,0)),IR,axes=(2,0))
    return new_Gs

def imps_to_fmps(Gs,SVMs,N):
    # return right-normalized fmps
    N=len(Gs); d=Gs[0].shape[1]
    theta=np.tensordot()
    X,S,Y=np.linalg.svd(theta,full_matrices=False)
    
    new_Gs=[None]*N
    for site in xrange(N):
        if site==0 or site==N-1:
            
        elif site<=N/2-1 and site!=0:
            
        elif site>N/2-1 and site!=N-1:
            
    return new_Gs
