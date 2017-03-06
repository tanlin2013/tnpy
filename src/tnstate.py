"""
This file contains the fundamental functions for the Matrix Product State (MPS) operations. 
"""

import numpy as np

class MPS:   
    def __init__(self,whichMPS,d,chi,**N):
        """
        * Parameters:
            * whichMPS: string, {'i','f'} 
                If whichMPS='i', an infinite MPS is initialized. Otherwise if whichMPS='f', a finite MPS is created.  
            * d: int
                The physical bond dimension which is associated to the dimension of single-particle Hilbert space.
            * chi: int
                The visual bond dimension to be keep after the Singular Value Decomposition (SVD).               
        """
        if whichMPS!='i' and whichMPS!='f':
            raise ValueError('Only iMPS and fMPS are supported.')
        self.whichMPS=whichMPS
        self.d=d
        self.chi=chi
        self.N=N
        
    def initialize_MPS(self,canonical_form='R'):
        """
        Randomly initialize the MPS.
    
        * Parameters:
            * N: int, optional
                If whichMPS='f', the size of system N is needed.           
            * canonical_form: string, {'L','R','GL'}, default='R'
                If whichMPS='f', fMPS can be represented as left-normalized, right-normalized or the standard (Gamma-Lambda representation) MPS.            
        
        * Returns: 
            * Gs: list of ndarray
                A list of rank-3 tensors which represents the MPS. The order of tensor is (chi,d,chi) or (d,chi) for the boundaries if fMPS is considered.  
            * SVMs: list of ndarray
                A list of singular value matrices. SVMs is always return for iMPS. But, for the fMPS SVMs only return when canonical_form='GL'.
        """
        
        # Check the input variables
        if self.whichMPS=='f': 
            if not canonical_form in ['L','R','GL'] and :
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
            Gs=[None]*self.N ; SVMs=[None]*self.N ; N_parity=N%2
            for site in xrange(N):        
                if N_parity==0:
                    if site==0 or site==N-1:
                        Gs.append(np.random.rand(self.d,min(self.d,self.chi)))
                    elif site<=N/2-1 and site!=0:
                        Gs.append(np.random.rand(min(self.d**site,self.chi),self.d,min(self.d**(site+1),self.chi)))
                    elif site>N/2-1 and site!=N-1:
                        Gs.append(np.random.rand(min(self.d**(N-site),self.chi),self.d,min(self.d**(N-1-site),self.chi)))
                elif N_parity==1:
                    if site==0 or site==N-1:
                        Gs.append(np.random.rand(self.d,min(self.d,self.chi)))
                    elif site<N/2 and site!=0:
                        Gs.append(np.random.rand(min(self.d**site,self.chi),self.d,min(self.d**(site+1),self.chi)))
                    elif site==N/2:
                        Gs.append(np.random.rand(min(self.d**site,self.chi),self.d,min(self.d**site,self.chi)))
                    elif site>N/2 and site!=N-1:
                        Gs.append(np.random.rand(min(self.d**(N-site),self.chi),self.d,min(self.d**(N-1-site),self.chi)))        
            # Canonical normalization of the MPS
            if canonical_form=='L':
                Gs=self.normalize_MPS(Gs,order='L')
                return Gs
            elif canonical_form=='R':
                Gs=self.normalize_MPS(Gs,order='R')
                return Gs
            elif canonical_form=='GL':
                Gs,SVMs=self.normalize_MPS(Gs,order='GL')
                return Gs,SVMs          
    
    def normalize_MPS(self,Gs,order):
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
        N=len(Gs) ; SVMs=[None]*(N-1)
        for site in xrange(N-1):
            if site==0:
                theta=Gs[site]
            else:
                theta=np.ndarray.reshape(Gs[site],(self.d*Gs[site].shape[0],Gs[site].shape[2]))
            X,S,Y=np.linalg.svd(theta,full_matrices=False)
            SVMs[site]=np.diagflat(S/np.linalg.norm(S))
            if site==N-2:
                Gs[site+1]=np.tensordot(Gs[site+1],Y,axes=(1,1))
            else:
                Gs[site+1]=np.tensordot(Y,Gs[site+1],axes=(1,0))
            if site==0:
                Gs[site]=np.ndarray.reshape(X,(self.d,Gs[site].shape[1]))
            else:
                Gs[site]=np.ndarray.reshape(X,(Gs[site].shape[0],self.d,Gs[site].shape[2]))
        if order=='L':
            for site in xrange(1,N):
                if site==N-1:
                    Gs[site]=np.tensordot(SVMs[site-1],Gs[site],axes=(1,1))
                else:
                    Gs[site]=np.tensordot(SVMs[site-1],Gs[site],axes=(1,0))
            return Gs
        elif order=='R':
            """
            for site in xrange(N-1):
                if site==0:
                    Gs[site]=np.tensordot(Gs[site],SVMs[site],axes=(1,0))
                else:
                    Gs[site]=np.tensordot(Gs[site],SVMs[site],axes=(2,0))
            """
            for site in range(N-1,0,-1):
                if site==N-1:
                    theta=Gs[site]
                else:
                    theta=np.ndarray.reshape(Gs[site],(Gs[site].shape[0],self.d*Gs[site].shape[2]))      
                X,S,Y=np.linalg.svd(theta,full_matrices=False)                
                if site==1:
                    Gs[site-1]=np.tensordot(Gs[site-1],np.dot(X,np.diagflat(S/np.linalg.norm(S))),axes=(1,0))
                else:         
                    Gs[site-1]=np.tensordot(Gs[site-1],np.dot(X,np.diagflat(S/np.linalg.norm(S))),axes=(2,0))
                if site==N-1:
                    Gs[site]=np.ndarray.reshape(Y,(self.d,Gs[site].shape[1]))
                else:
                    Gs[site]=np.ndarray.reshape(Y,(Gs[site].shape[0],self.d,Gs[site].shape[2]))  
            return Gs
        elif order=='GL':
            return Gs,SVMs
        else:
            raise ValueError('The order must be either L or R.')

def transfer_operator(Gs,A):
    if Gs.ndim==2:
        trans=np.tensordot(Gs,np.tensordot(A,np.conjugate(Gs),axes=(2,0)),axes=(0,0))
    else:                        
        trans=np.tensordot(Gs,np.tensordot(A,np.conjugate(Gs),axes=(2,1)),axes=(1,0))
        trans=np.swapaxes(trans,1,2)
        trans=np.swapaxes(trans,3,4)
        trans=np.swapaxes(trans,2,3)
    return trans    
