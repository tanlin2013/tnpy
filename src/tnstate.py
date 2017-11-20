"""
This file contains the fundamental functions for the Matrix Product State (MPS) operations. 
"""

import numpy as np

class MPS:   
    def __init__(self, whichMPS, d, chi, N=None):
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
        if whichMPS != 'i' and whichMPS != 'f':
            raise ValueError('Only iMPS and fMPS are supported.')
        self.whichMPS = whichMPS
        self.d = d
        self.chi = chi
        self.N = N
        
    def initialize(self, canonical_form='R'):
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
        if self.whichMPS == 'f': 
            if not canonical_form in ['L','R'] or type(self.N) is not int:
                raise ValueError('canonical_form and size must be specified when whichMPS=f.')        
        
        if self.whichMPS == 'i':
            # Create the iMPS
            Gs = [None]*2; SVMs = [None]*2
            for site in xrange(2):
                Gs[site] = np.random.rand(self.chi,self.d,self.chi)
                SVMs[site] = np.diagflat(np.random.rand(self.chi))
            return Gs, SVMs    
        elif self.whichMPS == 'f':
            # Create the fMPS
            Gs = [None]*self.N; N_parity = self.N%2
            for site in xrange(self.N):        
                if N_parity == 0:
                    if site == 0 or site == self.N-1:
                        Gs[site] = np.random.rand(self.d,min(self.d,self.chi))
                    elif site <= self.N/2-1 and site != 0:
                        Gs[site] = np.random.rand(min(self.d**site,self.chi),self.d,min(self.d**(site+1),self.chi))
                    elif site > self.N/2-1 and site != self.N-1:
                        Gs[site] = np.random.rand(min(self.d**(self.N-site),self.chi),self.d,min(self.d**(self.N-1-site),self.chi))
                elif N_parity == 1:
                    if site == 0 or site == self.N-1:
                        Gs[site] = np.random.rand(self.d,min(self.d,self.chi))
                    elif site < self.N/2 and site != 0:
                        Gs[site] = np.random.rand(min(self.d**site,self.chi),self.d,min(self.d**(site+1),self.chi))
                    elif site == self.N/2:
                        Gs[site] = np.random.rand(min(self.d**site,self.chi),self.d,min(self.d**site,self.chi))
                    elif site > self.N/2 and site != self.N-1:
                        Gs[site] = np.random.rand(min(self.d**(self.N-site),self.chi),self.d,min(self.d**(self.N-1-site),self.chi))        
            # Canonical normalization of the MPS
            if canonical_form == 'L':
                Gs = normalize_fmps(Gs,order='L')
            elif canonical_form == 'R':
                Gs = normalize_fmps(Gs,order='R')
            return Gs       

def _normalize_fmps(Gs, order, site):
    N = len(Gs); d = Gs[0].shape[0]
    if order == 'L':
        if site == 0:    
            theta = Gs[site]
        else:
            theta = np.ndarray.reshape(Gs[site],(d*Gs[site].shape[0],Gs[site].shape[2]))     
        X, S, Y = np.linalg.svd(theta,full_matrices=False)                
        if site == N-2:
            Gs[site+1] = np.tensordot(Gs[site+1],np.dot(np.diagflat(S),Y),axes=(1,1))
        else:
            Gs[site+1] = np.tensordot(np.dot(np.diagflat(S),Y),Gs[site+1],axes=(1,0))
        if site == 0:
            Gs[site] = X
        else:
            Gs[site] = np.ndarray.reshape(X,Gs[site].shape)
    elif order == 'R':
        if site == N-1:      
            theta = np.transpose(Gs[site])
        else:    
            theta = np.ndarray.reshape(Gs[site],(Gs[site].shape[0],d*Gs[site].shape[2]))     
        X, S, Y = np.linalg.svd(theta,full_matrices=False)                
        if site == 1:
            Gs[site-1] = np.tensordot(Gs[site-1],np.dot(X,np.diagflat(S)),axes=(1,0))
        else:         
            Gs[site-1] = np.tensordot(Gs[site-1],np.dot(X,np.diagflat(S)),axes=(2,0))
        if site == N-1:
            Gs[site] = np.transpose(Y)
        else:
            Gs[site] = np.ndarray.reshape(Y,Gs[site].shape)
    return Gs        
        
def normalize_fmps(Gs, order):
    """
    Canonical normalization of the fMPS.
        
    * Parameters:
        * Gs: list of ndarray
            The fMPS wants to be left- or right-normalized.  
        * order: string, {'L','R','mix','GL'}
            Specified the direction of normalization.
    * Returns:
        * Gs: list of ndarray
            Left- or right-normalized MPS.
    """
    N = len(Gs); d = Gs[0].shape[0]
    if order == 'L':
        for site in xrange(N-1):
            Gs = _normalize_fmps(Gs, order, site)
        return Gs
    elif order == 'R':
        for site in range(N-1,0,-1):
            Gs = _normalize_fmps(Gs, order, site)
        return Gs
    elif order == 'mix':
        for site in xrange(N/2):
            Gs = _normalize_fmps(Gs, 'L', site)
        for site in xrange(N-1,N/2,-1):
            Gs = _normalize_fmps(Gs, 'R', site)
        theta = np.ndarray.reshape(Gs[N/2],(Gs[N/2].shape[0],d*Gs[N/2].shape[2]))
        X, S, Y = np.linalg.svd(theta,full_matrices=False)
        Gs[N/2-1] = np.tensordot(Gs[N/2-1],X,axes=(2,0))
        Gs[N/2] = np.ndarray.reshape(Y,Gs[N/2].shape)
        SVM = np.diagflat(S)
        return Gs, SVM
    #elif order == 'GL':
    #    return Gs, SVMs
    else:
        raise ValueError('The order must be L, R or mix.')

def fmps_norm(Gs):
    N = len(Gs); order = get_fmps_order(Gs)
    
    if order == 'R':
        for site in xrange(N):
            if site == 0:
                norm = np.tensordot(Gs[site],np.conjugate(Gs[site]),axes=(0,0))
            elif site == N-1:
                norm = np.tensordot(np.tensordot(norm,Gs[site],axes=(0,1)),np.conjugate(Gs[site]),axes=([0,1],[1,0]))
            else:
                norm = np.tensordot(np.tensordot(norm,Gs[site],axes=(0,0)),np.conjugate(Gs[site]),axes=([0,1],[0,1]))
    elif order == 'L':
        for site in xrange(N-1,-1,-1):
            if site == 0:
                norm = np.tensordot(np.tensordot(norm,Gs[site],axes=(0,1)),np.conjugate(Gs[site]),axes=([0,1],[1,0]))
            elif site == N-1:
                norm = np.tensordot(Gs[site],np.conjugate(Gs[site]),axes=(0,0))
            else:
                norm = np.tensordot(np.tensordot(norm,Gs[site],axes=(0,2)),np.conjugate(Gs[site]),axes=([0,2],[2,1]))
    return norm        
        
def get_fmps_order(Gs):
    N=len(Gs); chi=Gs[0].shape[1]
    def G2(site):
        G2 = np.tensordot(Gs[site],np.conjugate(Gs[site]),axes=(0,0))
        return G2
    if np.allclose(G2(0),np.identity(chi),atol=1e-12): order = 'L'
    elif np.allclose(G2(N-1),np.identity(chi),atol=1e-12): order = 'R'    
    return order        
        
def transfer_operator(G, M):
    if G.ndim == 2:
        trans = np.tensordot(G,np.tensordot(M,np.conjugate(G),axes=(2,0)),axes=(0,0))
    else:                        
        trans = np.tensordot(G,np.tensordot(M,np.conjugate(G),axes=(2,1)),axes=(1,0))
        trans = np.swapaxes(trans,1,2)
        trans = np.swapaxes(trans,3,4)
        trans = np.swapaxes(trans,2,3)
    return trans    

def increase_bond_dim(Gs, old_chi, new_chi):
    N = len(Gs); d=Gs[0].shape[0]
    new_Gs = [None]*N
    for site in xrange(N):
        if site == 0 or site == N-1:
            I = np.eye(min(d,old_chi),min(d,new_chi))
            new_Gs[site] = np.tensordot(Gs[site],I,axes=(1,0))
        elif site <= N/2-1 and site != 0:
            IL = np.eye(min(d**site,old_chi),min(d**site,new_chi))
            IR = np.eye(min(d**(site+1),old_chi),min(d**(site+1),new_chi))
            new_Gs[site] = np.tensordot(np.tensordot(IL,Gs[site],axes=(0,0)),IR,axes=(2,0))
        elif site > N/2-1 and site != N-1:
            IL = np.eye(min(d**(N-site),old_chi),min(d**(N-site),new_chi))
            IR = np.eye(min(d**(N-1-site),old_chi),min(d**(N-1-site),new_chi))
            new_Gs[site] = np.tensordot(np.tensordot(IL,Gs[site],axes=(0,0)),IR,axes=(2,0))
    return new_Gs

def imps_to_fmps(Gs, SVMs, N):
    # return right-normalized fmps
    d = Gs[0].shape[1]; chi = Gs[0].shape[0]    
    new_Gs = [None]*N
    for site in xrange(N):
        G = np.tensordot(Gs[site%2],SVMs[site%2],axes=(2,0))
        if site == 0 or site == N-1:
            IL = np.eye(chi,1)
            IR = np.eye(chi,min(d,chi))
            new_Gs[site] = np.tensordot(np.tensordot(IL,G,axes=(0,0)),IR,axes=(2,0))
            new_Gs[site] = np.reshape(new_Gs[site],(d,min(d,chi)))
        elif site <= N/2-1 and site != 0:
            IL = np.eye(chi,min(d**site,chi))
            IR = np.eye(chi,min(d**(site+1),chi))
            new_Gs[site] = np.tensordot(np.tensordot(IL,G,axes=(0,0)),IR,axes=(2,0))
        elif site > N/2-1 and site != N-1:
            IL = np.eye(chi,min(d**(N-site),chi))
            IR = np.eye(chi,min(d**(N-1-site),chi))
            new_Gs[site] = np.tensordot(np.tensordot(IL,G,axes=(0,0)),IR,axes=(2,0))
    return new_Gs

def lengthen_fmps(Gs, new_N):
    old_N = len(Gs); new_Gs = np.copy(Gs).tolist()
    G = [Gs[old_N/2-1], Gs[old_N/2]]
    k = 0
    for half_length in xrange(old_N/2,new_N-old_N/2):
        parity = (k+1)%2
        new_Gs.insert(half_length, G[parity])
        k += 1
    return new_Gs
